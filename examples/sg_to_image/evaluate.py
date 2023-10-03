import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional, Union, List
import itertools
import json

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from utils import ObjectDetectionMetrics, compute_average_precision

from utils import ListEvalDataset

import torch_fidelity
from simsg import SGModel


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--depth_column", type=str, default="depth", help="The column of the dataset containing the depth map."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="objects_str",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--triplets_column",
        type=str,
        default="triplets",
        help="The column of the dataset containing the list of triplets",
    )
    parser.add_argument(
        "--boxes_column",
        type=str,
        default="boxes",
        help="The column of the dataset containing the list of bounding boxes of the objects",
    )
    parser.add_argument(
        "--objects_column",
        type=str,
        default="objects",
        help="The column of the dataset containing the list of the objects in the image",
    )
    parser.add_argument(
        "--cond_place",
        type=str,
        default="attn",
        choices=["latent", "attn"],
        help="Where to append the condition",
    )
    parser.add_argument(
        "--const_train_prompt", type=str, default="",
        help="A prompt to be used constantly for training as a prefix to the caption."
    )
    parser.add_argument(
        "--caption_type",
        type=str,
        default="none",
        choices=["none", "const", "triplets", "objects"],
        help=(
            "Whether to disable captions during training. Check if you have something else to condition on, "
            "like scene graphs. "
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help=(
            "Skip generation and only calculate metrics."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help=(
            "Number of samples to generate at once."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help=(
            "Checkpoint number to load from."
        ),
    )
    parser.add_argument("--vocab_json", type=str, default="mnt/students/vocab.json", help="The path to the vocab file.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--lora_rank", type=int, default=4, help="The rank of LoRA to use.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--shuffle_triplets", action="store_true", help="Whether or not to shuffle the triplets in the training set."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--sg_type",
        type=str,
        default='simsg',
        choices=('simsg', 'sgnet'),
        help="Type of sg encoder.",
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def get_vg_config(split):
    return {'vocab': '/mnt/datasets/visual_genome/vocab.json',
            'h5_path': f'/mnt/datasets/visual_genome/{split}.h5',
            'image_dir': '/mnt/datasets/visual_genome/images'}


vg_configs = {k: get_vg_config(k) for k in ['train', 'val', 'test']}


def set_lora_layers(unet):
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                                  rank=4)

    unet.set_attn_processor(lora_attn_procs)

    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

    lora_layers_ = AttnProcsLayers(unet.attn_processors)

    return lora_layers_


def main():
    args = parse_args()

    image_column = args.image_column
    depth_column = args.depth_column
    caption_column = args.caption_column
    triplets_column = args.triplets_column
    boxes_column = args.boxes_column
    objects_column = args.objects_column

    MODEL_ID = 'stabilityai/stable-diffusion-2'
    MODEL_PATH = args.model_path
    PATH = f'/mnt/workfiles/exp/{MODEL_PATH}'

    if args.checkpoint is not None:
        PATH = os.path.join(PATH, f'checkpoint-{args.checkpoint}')
    else:
        dirs = os.listdir(PATH)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        PATH = os.path.join(PATH, path)

    OUTPUT_DIR = f'/mnt/workfiles/gens/{MODEL_PATH}'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    BSZ = args.batch_size
    RESOLUTION = args.resolution
    CENTER_CROP = True if '-cc' in MODEL_PATH else False
    dataset_type = 'clevr' if 'clevr' in PATH else 'vg'
    if dataset_type == 'clevr':
        vocab_json = '/mnt/workfiles/diffusers/examples/sg_to_image/vocab.json'
        DATASET_NAME = 'erkam/clevr-full-v5'
    else:
        vocab_json = '/mnt/datasets/visual_genome/vocab.json'
        DATASET_NAME = 'vg'

    device = torch.device('cuda')

    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")

    unet = unet.requires_grad_(False)
    text_encoder = text_encoder.requires_grad_(False)
    unet = unet.to(device, dtype=weight_dtype)
    text_encoder = text_encoder.to(device, dtype=weight_dtype)
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    lora_layers = set_lora_layers(unet)
    lora_layers.load_state_dict(torch.load(f'{PATH}/pytorch_lora_weights.bin'))

    with open(vocab_json, 'r') as f:
        vocab = json.load(f)

    kwargs = {
        'vocab': vocab,
        'embedding_dim': 1024,
        'gconv_dim': 1024,
        'gconv_hidden_dim': 512,
        'gconv_num_layers': 5,
        'feats_in_gcn': False,
        'feats_out_gcn': True,
        'is_baseline': False,
        'is_supervised': True,
        'tokenizer': tokenizer,
        'text_encoder': text_encoder,
        'identity': True,
        'reverse_triplets': False,
    }
    sg_net = SGModel(**kwargs)
    sg_net.load_state_dict(torch.load(f'{PATH}/sg_encoder.pt'))
    sg_net.to(device, dtype=weight_dtype)
    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        unet=accelerator.unwrap_model(unet),
        torch_dtype=torch.float32,
    )

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    sg_net.eval()

    # sg_p = sum(p.numel() for p in sg_net.parameters())
    # lora_p = sum(p.numel() for p in lora_layers.parameters())
    # print(f'SGNet total parameters {sg_p / 1e6}M')
    # print(f'LoRA total parameters {lora_p / 1e6}M')

    def prepare_sg_embeds(examples, is_train=True):
        if dataset_type == 'clevr':
            max_length = (8, 21)
        else:
            max_length = (30, 21)

        num_objs_per_image = []
        num_triplets_per_image = []
        obj_offset = 0
        all_objects = []
        all_triplets = []
        all_boxes = []
        for triplets, boxes, objects in zip(examples[triplets_column], examples[boxes_column],
                                            examples[objects_column]):
            num_objs, num_triplets = objects.shape[0], triplets.shape[0]

            tr = triplets.clone()
            tr[:, 0] += obj_offset
            tr[:, 2] += obj_offset

            all_triplets.append(tr)
            all_boxes.append(boxes)
            all_objects.append(objects)

            num_objs_per_image.append(num_objs)
            num_triplets_per_image.append(num_triplets)

            obj_offset += num_objs

        all_objects = torch.cat(all_objects, dim=0).to(device)
        all_triplets = torch.cat(all_triplets, dim=0).to(device)
        all_boxes = torch.cat(all_boxes, dim=0).to(device)
        with torch.no_grad():
            embed = sg_net(all_triplets, all_objects, all_boxes, max_length=max_length,
                           batch_size=BSZ)

        embeds = torch.split(embed, num_objs_per_image, dim=0)

        # pad embeds to max length
        embeds = [F.pad(e, (0, 0, 0, max_length[0] - e.shape[0])) for e in embeds]
        embeds = torch.stack(embeds, dim=0)

        assert embeds.shape == (len(examples[objects_column]), max_length[0], 1024)

        return embeds

    val_transforms = transforms.Compose(
        [
            transforms.Resize(RESOLUTION,
                              interpolation=transforms.InterpolationMode.BILINEAR) if CENTER_CROP else transforms.Resize(
                (RESOLUTION, RESOLUTION), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(RESOLUTION) if CENTER_CROP else transforms.Lambda(lambda x: x)
        ]
    )

    def scale_box(boxes):
        assert dataset_type == 'clevr', 'Only CLEVR dataset needs box scaling!'
        if isinstance(boxes, list):
            boxes = torch.tensor(boxes)
        boxes = boxes.to(device)

        if not CENTER_CROP:
            return boxes

        boxes = boxes * torch.tensor([320, 240, 320, 240]).to(device) - torch.tensor(
            [40, 0, 40, 0]).to(device)
        boxes = boxes.clip(0, 240) / 240
        return boxes

    def tokenize_captions(examples, is_train=True):
        return torch.zeros((len(examples[caption_column]), 77), device=device)

    def preprocess_val(examples):
        assert dataset_type == 'clevr', 'Only CLEVR dataset needs preprocess_val!'
        examples[image_column] = [val_transforms(image.convert("RGB")) for image in
                                  examples[image_column]]
        examples[triplets_column] = [torch.tensor(triplets, device=device, dtype=torch.long) for triplets in
                                     examples[triplets_column]]
        examples[boxes_column] = [scale_box(boxes) for boxes in examples[boxes_column]]
        examples[objects_column] = [torch.tensor(objects, device=device, dtype=torch.long) for objects in
                                    examples[objects_column]]
        examples["input_ids"] = tokenize_captions(examples, is_train=False)
        examples["sg_embeds"] = prepare_sg_embeds(examples, is_train=False)
        return examples

    def val_collate_fn(examples):
        assert dataset_type == 'clevr', 'Only CLEVR dataset needs collate_fn!'
        all_images = [example[image_column] for example in examples]

        input_ids = torch.stack([example["input_ids"] for example in examples])
        sg_embeds = torch.stack([example["sg_embeds"] for example in examples])

        boxes = [example[boxes_column] for example in examples]
        objects = [example[objects_column] for example in examples]
        triplets = [example[triplets_column] for example in examples]
        out = {
            "image": all_images,
            "input_ids": input_ids,
            "sg_embeds": sg_embeds,
            "boxes": boxes,
            "objects": objects,
            "triplets": triplets,
        }
        return out

    # load the datasets
    if dataset_type == 'clevr':
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            DATASET_NAME,
            keep_in_memory=True
        )
        dataset = dataset['test'].with_transform(preprocess_val)
    elif dataset_type == 'vg':
        from simsg import VGDiffDatabase, get_collate_fn
        dataset = VGDiffDatabase(**vg_configs['test'],
                                 image_size=RESOLUTION,
                                 max_samples=None,
                                 use_depth=False)
        collate_fn = get_collate_fn(prepare_sg_embeds, tokenize_captions)
    else:
        raise ValueError(f"Dataset {DATASET_NAME} not supported. Supported datasets: clevr, vg")

    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        collate_fn=val_collate_fn if dataset_type == 'clevr' else collate_fn,
        batch_size=BSZ,
        num_workers=args.dataloader_num_workers,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)

    if not args.skip_generation:
        for step, batch in enumerate(tqdm(loader)):
            sg_embeds = batch["sg_embeds"].to(device)

            with accelerator.autocast():
                images = pipeline(prompt_embeds=sg_embeds,
                                  height=RESOLUTION, width=RESOLUTION, num_inference_steps=200,
                                  generator=generator).images
                print(len(images))
                for i, image in enumerate(images):
                    image.save(os.path.join(OUTPUT_DIR, f"{step * BSZ + i}.png"))

    metrics = torch_fidelity.calculate_metrics(
        input1=OUTPUT_DIR,
        input2=ListEvalDataset(dataset),
        cuda=True,
        isc=True,
        fid=True,
        kid=True,
        kid_subset_size=BSZ,
        verbose=True)

    # save metrics to txt
    with open(os.path.join(OUTPUT_DIR, 'metrics.txt'), 'w') as f:
        f.write(str(metrics))


if __name__ == "__main__":
    main()
