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
from examples.sg_to_image.train_sg_to_image_lora import parse_args
from utils import ObjectDetectionMetrics, compute_average_precision

from utils import ListDataset
from utils import interpolate

import torch_fidelity
from simsg import SGModel


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
    MODEL_PATH = 'clevr/sg2im-128-bs-32-cc'
    PATH = f'/mnt/workfiles/exp/{MODEL_PATH}'
    OUTPUT_DIR = f'/mnt/workfiles/gens/{MODEL_PATH}'
    BSZ = 32
    RESOLUTION = 256
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

    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")

    unet = unet.requires_grad_(False)
    text_encoder = text_encoder.requires_grad_(False)
    unet = unet.to(torch.device('cuda'), dtype=torch.float16)
    text_encoder = text_encoder.to(torch.device('cuda'), dtype=torch.float16)
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

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=accelerator.unwrap_model(unet),
        revision=args.revision,
        torch_dtype=torch.float32,
    )

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    sg_net.eval()

    sg_p = sum(p.numel() for p in sg_net.parameters())
    lora_p = sum(p.numel() for p in lora_layers.parameters())
    print(f'SGNet total parameters {sg_p / 1e6}M')
    print(f'LoRA total parameters {lora_p / 1e6}M')
    dataset_type = 'clevr' if 'clevr' in args.dataset_name else 'vg'

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

        if not args.center_crop:
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

    for step, batch in enumerate(loader):
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
        input2=dataset,
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
