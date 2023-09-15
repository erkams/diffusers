# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

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

from utils import ListDataset
import torch_fidelity

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")

logger = get_logger(__name__, log_level="INFO")

datasets.config.IN_MEMORY_MAX_SIZE = 3_000_000_000


def get_vg_config(split):
    return {'vocab': '/mnt/datasets/visual_genome/vocab.json',
            'h5_path': f'/mnt/datasets/visual_genome/{split}.h5',
            'image_dir': '/mnt/datasets/visual_genome/images'}


vg_configs = {k: get_vg_config(k) for k in ['train', 'val', 'test']}


def save_model_card(repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- sg-to-image
- scene-graph
- stable-diffusion
- stable-diffusion-diffusers
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)

    print(f"*** Model card saved in {repo_folder} ***")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--sg_model_path",
        type=str,
        default='/mnt/simsg/clevr_models/checkpoint.pt',
        help="Path to pretrained SG model",
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
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
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
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--num_eval_images",
        type=int,
        default=20,
        help="Number of images that should be generated during evaluation during training.",
    )
    parser.add_argument(
        "--num_eval_batches",
        type=int,
        default=20,
        help="Number of images that should be generated during evaluation during training.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=2,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
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
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate_sg",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_lora",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler_sg",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_scheduler_lora",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
             "More details here: https://arxiv.org/abs/2303.09556.",
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
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument(
        "--train_sg", action="store_true", help="Whether or not to train scene graph embedding network."
    )
    parser.add_argument(
        "--identity", action="store_true", help="Whether or not to add identity to the sg embed"
    )
    parser.add_argument(
        "--start_lora", type=int, default=500, help="Number of steps after to start the lora training."
    )
    parser.add_argument("--vocab_json", type=str, default="mnt/students/vocab.json", help="The path to the vocab file.")

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument('--leading_metric', type=str, default='FID', choices=('ISC', 'FID', 'KID', 'PPL'))
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
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--lora_rank", type=int, default=4, help="The rank of LoRA to use.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
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
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def square_imgs(img: Union[Image.Image, List[Image.Image]], size: int) -> Union[Image.Image, List[Image.Image]]:
    """
    Crops the image to a square and resizes it to the given size.
    Args:
        img: The image to crop and resize.
        size: The size to resize the image to.

    Returns:
        The cropped and resized image.
    """

    if isinstance(img, list):
        return [square_imgs(i, size) for i in img]
    width, height = img.size
    if width == height:
        return img.resize((size, size), Image.BILINEAR)
    elif width > height:
        diff = width - height
        img = img.crop((diff // 2, 0, diff // 2 + height, height))
        return img.resize((size, size), Image.BILINEAR)
    else:
        diff = height - width
        img = img.crop((0, diff // 2, width, diff // 2 + width))
        return img.resize((size, size), Image.BILINEAR)


def make_grid(imgs, rows: int, cols: int) -> Image.Image:
    """
    Creates a grid of images.
    Args:
        imgs: The images to put in the grid.
        rows: The number of rows in the grid.
        cols: The number of columns in the grid.

    Returns:
        The grid of images.
    """

    width, height = (64, 64)
    grid = Image.new("RGB", (cols * width, rows * height))
    for i, img in enumerate(imgs):
        # convert to PIL if it's a tensor
        if isinstance(img, torch.Tensor):
            image = transforms.ToPILImage()(img)
        else:
            image = img
        grid.paste(image.resize((width, height)), box=(width * (i % cols), height * (i // cols)))
    return grid


def build_sg_encoder(args, tokenizer=None, text_encoder=None):
    with open(args.vocab_json, 'r') as f:
        vocab = json.load(f)
    if args.sg_type == 'simsg':
        if args.train_sg:
            from simsg import SGModel
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
                'identity': args.identity,
            }
            sg_net = SGModel(**kwargs)
            sg_net.train()

        else:
            assert args.sg_model_path is not None, "Please specify a path to a pretrained SG model."

            from simsg import SIMSGModel
            # initializing the SG Model with the pretrained model
            checkpoint = torch.load(args.sg_model_path)
            sg_net = SIMSGModel(**checkpoint['model_kwargs'])
            sg_net.load_state_dict(checkpoint['model_state'])
            sg_net.enable_embedding(text_encoder=text_encoder, tokenizer=tokenizer)
            sg_net.image_size = 64
            sg_net.requires_grad_(False)
    elif args.sg_type == 'sgnet':
        sg_net = None
        pass
    else:
        raise ValueError(f"Unknown sg_type: {args.sg_type}! Please choose from ['simsg', 'sgnet'].")

    return sg_net


def main():
    args = parse_args()
    assert args.dataset_name is not None, "Need a dataset name."
    dataset_type = 'clevr' if 'clevr' in args.dataset_name else 'vg'

    image_column = args.image_column
    caption_column = args.caption_column
    triplets_column = args.triplets_column
    boxes_column = args.boxes_column
    objects_column = args.objects_column

    # logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        # logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Setup metrics
    leading_metric, last_best_metric, metric_greater_cmp = {
        'ISC': (torch_fidelity.KEY_METRIC_ISC_MEAN, 0.0, float.__gt__),
        'FID': (torch_fidelity.KEY_METRIC_FID, float('inf'), float.__lt__),
        'KID': (torch_fidelity.KEY_METRIC_KID_MEAN, float('inf'), float.__lt__),
        'PPL': (torch_fidelity.KEY_METRIC_PPL_MEAN, float('inf'), float.__lt__),
    }[args.leading_metric]

    isc_metric, last_best_isc, isc_cmp = torch_fidelity.KEY_METRIC_ISC_MEAN, 0.0, float.__gt__

    object_detection_metrics = ObjectDetectionMetrics(dataset_type=dataset_type)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    sg_net = build_sg_encoder(args)

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)

    text_encoder.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    sg_net.to(accelerator.device)
    # sg_net = accelerator.prepare(sg_net)

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    for param in params_to_freeze:
        param.requires_grad = False

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # Set correct lora layers
    def set_lora_layers():
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
                                                      rank=args.lora_rank)

        unet.set_attn_processor(lora_attn_procs)

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        lora_layers_ = AttnProcsLayers(unet.attn_processors)

        optimizer = optimizer_cls(
            lora_layers_.parameters(),
            lr=args.learning_rate_lora,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        lr_scheduler = get_scheduler(
            args.lr_scheduler_lora,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

        return lora_layers_, optimizer, lr_scheduler

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate_lora = (
                args.learning_rate_lora * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

        args.learning_rate_sg = (
                args.learning_rate_sg * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.train_sg:
        # optimizer = optimizer_cls(
        #     lora_layers.parameters(),
        #     lr=args.learning_rate,
        #     betas=(args.adam_beta1, args.adam_beta2),
        #     weight_decay=args.adam_weight_decay,
        #     eps=args.adam_epsilon,
        # )

        optimizer_sg = optimizer_cls(
            [*filter(lambda p: p.requires_grad, sg_net.parameters())],
            lr=args.learning_rate_sg,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        logger.info("train_sg is false. Only LoRA layers are being trained.")
        args.start_lora = 0
        # optimizer = optimizer_cls(
        #     lora_layers.parameters(),
        #     lr=args.learning_rate,
        #     betas=(args.adam_beta1, args.adam_beta2),
        #     weight_decay=args.adam_weight_decay,
        #     eps=args.adam_epsilon,
        # )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.

    def prepare_sg_embeds(examples, is_train=True):
        max_length = (8, 21)
        sg_embeds = []
        for triplets, boxes, objects in zip(examples[triplets_column], examples[boxes_column],
                                            examples[objects_column]):

            triplets = triplets.to(accelerator.device)
            boxes = boxes.to(accelerator.device)
            objects = objects.to(accelerator.device)

            if is_train:
                random.shuffle(triplets)
            if args.train_sg:
                embed = sg_net(triplets, objects, boxes, max_length=max_length, batch_size=args.train_batch_size)
            else:
                embed = sg_net.encode_sg(triplets, objects, boxes, max_length=max_length,
                                         batch_size=args.train_batch_size)

            sg_embeds.append(embed)

        sg_embed = torch.stack(sg_embeds, dim=0)
        if args.cond_place == 'latent':
            # here 2 is an hyperparameter
            out_shape = (1, 2, args.resolution / vae.config.scaling_factor, args.resolution / vae.config.scaling_factor)
            # resize vector with interpolation
            sg_embed = F.interpolate(sg_embed, size=(out_shape[-2], out_shape[-1]), mode='bilinear',
                                     align_corners=False)
            sg_embed = sg_embed.squeeze(0)
            sg_embed = torch.cat([sg_embed] * out_shape[1], dim=0)

        elif args.cond_place == 'attn':
            out_shape = (1, sum(max_length), 1024)
            # zeros = torch.zeros_like(sg_embed)
            # pad = torch.cat([zeros] * 3, dim=-1) # hardcoded for 3 now = out_shape[2]//sg_embed.shape[2] - 1
            # sg_embed = torch.cat([sg_embed, pad], dim=-1)
            # sg_embed = sg_embed.repeat(1, 1, out_shape[2]//sg_embed.shape[2])

        return sg_embed

    def get_caption(caption):
        if args.caption_type == 'const':
            return args.const_train_prompt
        elif args.caption_type == 'objects':
            objects = list(set(caption.replace(' is right of', ',')
                               .replace(' is left of', ',')
                               .replace(' is front of', ',')
                               .replace(' is behind', ',')
                               .split(', ')))
            return ", ".join(objects)
        elif args.caption_type == 'triplets':
            return caption

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        if args.caption_type == 'none':
            return torch.zeros((len(examples[caption_column]), 77), device=accelerator.device)
        captions = []
        for caption in examples[caption_column]:
            caption = get_caption(caption)
            if isinstance(caption, str):
                if is_train and args.shuffle_triplets:
                    triplets = caption.split(', ')
                    caption = ", ".join(random.sample(triplets, len(triplets)))
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def scale_box(boxes):
        if not args.center_crop:
            return boxes
        assert dataset_type == 'clevr', 'Only CLEVR dataset needs box scaling!'
        if isinstance(boxes, list):
            boxes = torch.tensor(boxes)
        boxes = boxes.to(accelerator.device)
        boxes = boxes * torch.tensor([320, 240, 320, 240]).to(accelerator.device) - torch.tensor(
            [40, 0, 40, 0]).to(accelerator.device)
        boxes = boxes.clip(0, 240) / 240
        return boxes

    def preprocess_train(examples):
        assert dataset_type == 'clevr', 'Only CLEVR dataset needs preprocess_train!'
        images = [image.convert("RGB") for image in examples[image_column]]
        examples[triplets_column] = [torch.tensor(triplets, device=accelerator.device, dtype=torch.long) for triplets in
                                     examples[triplets_column]]
        examples[boxes_column] = [scale_box(boxes) for boxes in examples[boxes_column]]
        examples[objects_column] = [torch.tensor(objects, device=accelerator.device, dtype=torch.long) for objects in
                                    examples[objects_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        examples["sg_embeds"] = prepare_sg_embeds(examples)
        return examples

    def preprocess_val(examples):
        assert dataset_type == 'clevr', 'Only CLEVR dataset needs preprocess_val!'
        examples[image_column] = [image.convert("RGB").resize((args.resolution, args.resolution)) for image in examples[image_column]]
        examples[triplets_column] = [torch.tensor(triplets, device=accelerator.device, dtype=torch.long) for triplets in
                                     examples[triplets_column]]
        examples[boxes_column] = [scale_box(boxes) for boxes in examples[boxes_column]]
        examples[objects_column] = [torch.tensor(objects, device=accelerator.device, dtype=torch.long) for objects in
                                    examples[objects_column]]
        examples["input_ids"] = tokenize_captions(examples, is_train=False)
        examples["sg_embeds"] = prepare_sg_embeds(examples, is_train=False)
        return examples

    def collate_fn(examples):
        assert dataset_type == 'clevr', 'Only CLEVR dataset needs collate_fn!'
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        sg_embeds = torch.stack([example["sg_embeds"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids, "sg_embeds": sg_embeds,
                "prompt": examples[0][caption_column],
                "triplets": examples[0][triplets_column], "boxes": examples[0][boxes_column]}

    if dataset_type == 'clevr':
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            keep_in_memory=True
        )
    elif dataset_type == 'vg':
        from simsg import VGDiffDatabase, get_collate_fn
        dataset = {
            k: VGDiffDatabase(**vg_configs[k],
                              image_size=args.resolution,
                              prepare_sg_embeds=prepare_sg_embeds,
                              tokenize_captions=tokenize_captions,
                              max_samples=args.max_train_samples)
            for k in ['train', 'val', 'test']}
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported. Supported datasets: clevr, vg")

    with accelerator.main_process_first():
        if dataset_type == "clevr":
            if args.max_train_samples is not None:
                dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess_train)
        elif dataset_type == "vg":
            train_dataset = dataset["train"]
            collate_fn = get_collate_fn(prepare_sg_embeds, tokenize_captions)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler_sg = get_scheduler(
        args.lr_scheduler_sg,
        optimizer=optimizer_sg,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    optimizer_sg, train_dataloader, lr_scheduler_sg = accelerator.prepare(
        optimizer_sg, train_dataloader, lr_scheduler_sg
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    def handle_hidden_states(input_ids=None, condition=None):
        if args.caption_type != 'none':
            encoder_hidden_states = text_encoder(input_ids)[0]
        if args.cond_place == 'attn':
            if args.caption_type != 'none':
                prompt_embeds = torch.cat((encoder_hidden_states, condition), dim=1)
            else:
                prompt_embeds = condition
        else:
            if args.caption_type != 'none':
                prompt_embeds = encoder_hidden_states
            else:
                ValueError('caption_type can\'t be none if cond_place is not attn')

        return prompt_embeds

    def validation_step(epoch):
        # Prepare validation sample
        val_sample = dataset['val'][0]
        val_sample[triplets_column] = [torch.tensor(val_sample[triplets_column], device=accelerator.device,
                                                    dtype=torch.long)]
        val_sample[objects_column] = [torch.tensor(val_sample[objects_column], device=accelerator.device,
                                                   dtype=torch.long)]
        val_sample[caption_column] = [val_sample[caption_column]]
        if dataset_type == 'clevr':
            val_sample[boxes_column] = [scale_box(val_sample[boxes_column])]
        else:
            val_sample[boxes_column] = [torch.tensor(val_sample[boxes_column], device=accelerator.device)]
        val_sample['sg_embeds'] = prepare_sg_embeds(val_sample)
        val_sample["input_ids"] = tokenize_captions(val_sample)

        validation_prompt = get_caption(val_sample[caption_column][0])


        logger.info(
            f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
            f" {validation_prompt}."
        )
        # create pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            revision=args.revision,
            torch_dtype=weight_dtype,
        )
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        # run inference
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        generated_images = []

        for _ in range(args.num_validation_images):
            generated_images.append(
                pipeline(prompt_embeds=
                         handle_hidden_states(input_ids=val_sample["input_ids"], condition=val_sample["sg_embeds"]),
                         height=args.resolution,
                         width=args.resolution,
                         num_inference_steps=30,
                         generator=generator).images[0]
            )

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in generated_images])
                tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
            if tracker.name == "wandb":
                img_gt = val_sample[image_column]
                # convert to PIL image
                if isinstance(img_gt, torch.Tensor):  # VG
                    img_gt = transforms.ToPILImage()(img_gt)
                elif isinstance(img_gt, list):  # CLEVR?
                    img_gt = img_gt[0]

                sizes = img_gt.size
                # calculate the crop box for center crop
                w = sizes[0]
                h = sizes[1]
                if w > h:
                    img_gt = img_gt.crop(((w - h) // 2, 0, (w + h) // 2, h))
                elif h > w:
                    img_gt = img_gt.crop((0, (h - w) // 2, w, (h + w) // 2))

                img_gt = img_gt.resize((args.resolution, args.resolution))

                img_logs = [
                    wandb.Image(image, caption=f"{i}: {validation_prompt}")
                    for i, image in enumerate(generated_images)
                ]
                img_logs = [wandb.Image(img_gt, caption=f"Ground truth")] + img_logs
                tracker.log({"validation": img_logs})

    def evaluation_step(global_step, test=False):
        nonlocal last_best_metric
        nonlocal last_best_isc

        num_batches = args.num_eval_batches if test else 1

        # Calculate FID metric on a subset of validation images
        logger.info(
            f"Running evaluation... \n Generating {num_batches} x {args.num_eval_images} images"
        )

        # create pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            revision=args.revision,
            torch_dtype=weight_dtype,
        )
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        # run inference
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        if test:
            dset = dataset['test']
        else:
            dset = dataset['val']

        if dataset_type == "clevr":
            dset = dset.with_transform(preprocess_val)

        loader = torch.utils.data.DataLoader(
            dset,
            shuffle=True if test else False,
            collate_fn=collate_fn,
            batch_size=args.num_eval_images,
            num_workers=args.dataloader_num_workers,
        )
        eval_samples = next(iter(loader))
        # eval_samples = dataset['val'].with_transform(preprocess_val)[:args.num_eval_images]

        is_vals = []
        fid_vals = []

        for _ in range(num_batches):
            eval_samples = next(iter(loader))
            images = []
            for input_ids, sg_embeds in zip(eval_samples["input_ids"], eval_samples["sg_embeds"]):
                # input_ids = sample['input_ids']
                # sg_embeds = sample['sg_embeds']
                # print(f'input id shape: {input_ids.shape}')
                # print(f'sg_embeds shape: {sg_embeds.shape}')
                input_ids = input_ids.unsqueeze(0).to(accelerator.device)
                sg_embeds = sg_embeds.unsqueeze(0).to(accelerator.device)
                with accelerator.autocast():
                    images.append(pipeline(prompt_embeds=handle_hidden_states(input_ids=input_ids, condition=sg_embeds),
                                           height=args.resolution, width=args.resolution, num_inference_steps=30,
                                           generator=generator).images[0])
            if not test:
                # BOX AND OBJECT DETECTION METRICS

                box_aps = 0.
                obj_aps = 0.
                sum_num_objs = 0.
                all_boxes_gt = eval_samples[boxes_column]
                all_objects_gt = eval_samples[objects_column]
                for i in range(len(images)):
                    boxes_gt = all_boxes_gt[i][:-1]

                    # boxes_gt = boxes_gt * torch.tensor([320, 240, 320, 240]).to(accelerator.device) - torch.tensor(
                    #     [40, 0, 40, 0]).to(accelerator.device)
                    # boxes_gt = boxes_gt.clip(0, 240) / 240

                    objects_gt = all_objects_gt[i][:-1]
                    ap_box, ap_obj, num_objs = object_detection_metrics.calculate(images[i], boxes_gt, objects_gt)
                    box_aps += ap_box
                    obj_aps += ap_obj
                    sum_num_objs += num_objs

                accelerator.log({"mAP_BOX": box_aps / len(images),
                                 "mAP_OBJ": obj_aps / len(images),
                                 "NUM_OBJS": sum_num_objs / len(images)}, step=global_step)

                logger.info("***** Eval results *****")
                logger.info(f"mAP_BOX: {box_aps / len(images)}")
                logger.info(f"mAP_OBJ: {obj_aps / len(images)}")
                logger.info(f"NUM_OBJS: {sum_num_objs / len(images)}")

                # VISUALIZE
                images_gt = eval_samples[image_column]
                if dataset_type == "clevr":
                    images_gt = square_imgs(images_gt, args.resolution)

                merged = []
                for i in range(len(images)):
                    merged.append(images_gt[i])
                    merged.append(images[i])

                grid = make_grid(merged, 5, 8)

                accelerator.log({"eval_images": [wandb.Image(grid, caption="Eval images")]})

            # FID AND IS METRICS
            images_gt = eval_samples[image_column]

            if dataset_type == "clevr":
                images_gt = square_imgs(images_gt, args.resolution)

            metrics = torch_fidelity.calculate_metrics(
                input1=ListDataset(images),
                input2=ListDataset(images_gt),
                cuda=True,
                isc=True,
                fid=True,
                kid=False,
                verbose=False)

            is_vals.append(metrics[isc_metric])
            fid_vals.append(metrics[leading_metric])

        metrics = {isc_metric: np.mean(is_vals), leading_metric: np.mean(fid_vals)}

        logger.info(f"*****{' Test' if test else ''} Eval results for STEP {global_step}*****")
        logger.info(f"ISC: {np.mean(is_vals)} +- {np.std(is_vals)}")
        logger.info(f"FID: {np.mean(fid_vals)} +- {np.std(fid_vals)}")

        if test:
            accelerator.log({"TEST_" + leading_metric: metrics[leading_metric],
                             "TEST_IS": metrics[isc_metric]},
                            step=global_step)
        else:
            accelerator.log({leading_metric: metrics[leading_metric], "IS": metrics[isc_metric]},
                            step=global_step)

            # save the generator if it improved
            if metric_greater_cmp(metrics[leading_metric], last_best_metric):
                logger.info(
                    f'Leading metric {leading_metric} improved from {last_best_metric} to {metrics[leading_metric]}')
                last_best_metric = metrics[leading_metric]
                torch.save(sg_net.state_dict(), f'./sg_encoder.pt')

            # save the generator if it improved
            if metric_greater_cmp(metrics[isc_metric], last_best_isc):
                logger.info(
                    f'Leading metric IS improved from {last_best_isc} to {metrics[isc_metric]}')
                last_best_isc = metrics[isc_metric]

    # torch.backends.cudnn.enabled = False
    # logger.info("***** Running eval check *****")
    # evaluation_step(0)
    # logger.info("***** Running validation check *****")
    # validation_step(0)
    # logger.info("***** Running test check *****")
    # evaluation_step(0, test=True)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    train_lora = False
    images = []

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    PIXEL_VALUES_KEY = 'pixel_values' if dataset_type == "clevr" else 'image'
    torch.save(sg_net.state_dict(), f'./sg_encoder.pt')
    for epoch in range(first_epoch, args.num_train_epochs):

        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if not train_lora and global_step + 1 >= args.start_lora:
                train_lora = True
                unet.train()
                logger.info('Starting LoRA training...')
                lora_layers, optimizer_lora, lr_scheduler_lora = set_lora_layers()
                lora_layers, optimizer_lora, lr_scheduler_lora = accelerator.prepare(lora_layers, optimizer_lora,
                                                                                     lr_scheduler_lora)

            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            if accelerator.is_main_process:
                if step == 0:
                    logger.info(f'STEP: {step}')
                    logger.info(f'sg embed shape: {batch["sg_embeds"].shape}')
                    logger.info(f'triplets: {batch[triplets_column]}')

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch[PIXEL_VALUES_KEY].to(dtype=weight_dtype)).latent_dist.sample()

                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if args.cond_place == 'latent':
                    noisy_latents = torch.cat([noisy_latents, batch['sg_embeds']], dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                with accelerator.autocast():
                    model_pred = unet(noisy_latents, timesteps, handle_hidden_states(input_ids=batch["input_ids"],
                                                                                     condition=batch['sg_embeds'])).sample

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps)
                    mse_loss_weights = (
                            torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if train_lora:
                    if accelerator.sync_gradients:
                        params_to_clip = lora_layers.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer_lora.step()
                    lr_scheduler_lora.step()
                    optimizer_lora.zero_grad()

                optimizer_sg.step()
                lr_scheduler_sg.step()
                optimizer_sg.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr_sg": lr_scheduler_sg.get_last_lr()[0]}
            if train_lora:
                logs["lr_lora"] = lr_scheduler_lora.get_last_lr()[0]
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if epoch % (8 * args.validation_epochs) == 0:
                print(f'***EVALUATION AT EPOCH {epoch}***')
                evaluation_step(global_step)

            if epoch % args.validation_epochs == 0:
                validation_step(epoch)

            if global_step % 5000 == 0 and global_step > 0:
                print(f'***EVALUATING ON TEST DATA AT EPOCH {epoch}***')
                evaluation_step(global_step, test=True)

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unet.save_attn_procs(args.output_dir)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images if images is not None else None,
                base_model=args.pretrained_model_name_or_path,
                dataset_name=args.dataset_name,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    # Final inference
    # val_sample = dataset['val'][0]
    # val_sample[triplets_column] = [torch.tensor(val_sample[triplets_column], device=accelerator.device)]
    # val_sample[boxes_column] = [torch.tensor(val_sample[boxes_column], device=accelerator.device)]
    # val_sample[objects_column] = [torch.tensor(val_sample[objects_column], device=accelerator.device)]
    # val_sample[caption_column] = [val_sample[caption_column]]
    # val_sample['sg_embeds'] = prepare_sg_embeds(val_sample)
    # val_sample["input_ids"] = tokenize_captions(val_sample).to(accelerator.device)

    # # Load previous pipeline
    # pipeline = StableDiffusionPipeline.from_pretrained(
    #     args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=weight_dtype
    # )
    # pipeline = pipeline.to(accelerator.device)

    # # load attention processors
    # pipeline.unet.load_attn_procs(args.output_dir)

    # encoder_hidden_states = text_encoder(val_sample["input_ids"])[0]

    # if args.cond_place == 'attn':
    #     prompt_embeds = torch.cat((encoder_hidden_states, val_sample['sg_embeds']), dim=1)
    # else:
    #     prompt_embeds = encoder_hidden_states

    # # run inference
    # generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    # images = []
    # for _ in range(args.num_validation_images):
    #     images.append(pipeline(prompt_embeds=prompt_embeds, num_inference_steps=30, generator=generator).images[0])

    # if accelerator.is_main_process:
    #     for tracker in accelerator.trackers:
    #         if tracker.name == "tensorboard":
    #             np_images = np.stack([np.asarray(img) for img in images])
    #             tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
    #         if tracker.name == "wandb":
    #             tracker.log(
    #                 {
    #                     "test": [
    #                         wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
    #                         for i, image in enumerate(images)
    #                     ]
    #                 }
    #             )

    accelerator.end_training()


if __name__ == "__main__":
    main()
