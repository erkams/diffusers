import argparse
import datetime
import json
from tqdm import tqdm
import os

import wandb

import torch
from torch import nn, optim

import datasets
from datasets import load_dataset
from torchvision import transforms

from diffusers import get_scheduler
from simsg import SGNet
import numpy as np

TRIPLETS = 'triplets'
OBJECTS = 'objects'
BOXES = 'boxes'
IMAGE = 'image'
DEPTH = 'depth'
DEPTH_LATENT = 'depth_latent'
IMAGE_LATENT = 'image_latent'

datasets.config.IN_MEMORY_MAX_SIZE = 3_000_000_000


def get_vg_config(split):
    return {'vocab': '/mnt/datasets/visual_genome/vocab.json',
            'h5_path': f'/mnt/datasets/visual_genome/{split}.h5',
            'image_dir': '/mnt/datasets/visual_genome/images'}


vg_configs = {k: get_vg_config(k) for k in ['train', 'val', 'test']}

SD_MODEL_ID = "stabilityai/stable-diffusion-2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='erkam/clevr-full-v6',
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
        "--output_dir",
        type=str,
        default="sg_encoder",
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
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
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
        "--max_val_samples",
        type=int,
        default=100,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
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
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
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
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--vocab_json", type=str, default="./vocab.json", help="The path to the vocab file.")

    args = parser.parse_args()
    return args


def build_model(args, device=None):
    with open(args.vocab_json, 'r') as f:
        vocab = json.load(f)

    sg_net = SGNet(vocab)
    sg_net.to(device)

    return sg_net


def build_dataloader(args, device=None):
    if 'clevr' in args.dataset_name:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            keep_in_memory=True
        )
    elif args.dataset_name == 'vg':
        raise NotImplementedError
    else:
        raise NotImplementedError

    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def collate_fn(examples):
        # pixel_values = torch.stack([example["pixel_values"] for example in examples])
        # pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        triplets = [example[TRIPLETS] for example in examples]
        objects = [example[OBJECTS] for example in examples]
        boxes = [example[BOXES] for example in examples]
        depth_latent = torch.stack([example[DEPTH_LATENT] for example in examples])
        image_latent = torch.stack([example[IMAGE_LATENT] for example in examples])
        return {
            # "pixel_values": pixel_values,
            TRIPLETS: triplets,
            BOXES: boxes,
            OBJECTS: objects,
            DEPTH_LATENT: depth_latent,
            IMAGE_LATENT: image_latent
        }

    def preprocess_train(examples):
        # images = [image.convert("RGB") for image in examples[IMAGE]]
        examples[TRIPLETS] = [torch.tensor(triplets, device=device, dtype=torch.long) for triplets in
                              examples[TRIPLETS]]
        examples[BOXES] = [boxes for boxes in examples[BOXES]]
        examples[OBJECTS] = [torch.tensor(objects, device=device, dtype=torch.long) for objects in
                             examples[OBJECTS]]
        examples[DEPTH_LATENT] = [torch.tensor(depth_latent, device=device, dtype=torch.float) for depth_latent in
                                  examples[DEPTH_LATENT]]
        examples[IMAGE_LATENT] = [torch.tensor(image_latent, device=device, dtype=torch.float) for image_latent in
                                  examples[IMAGE_LATENT]]
        # examples["pixel_values"] = [train_transforms(image) for image in images]
        return examples

    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        dataset["val"] = dataset["val"].shuffle(seed=args.seed).select(range(args.max_val_samples))

    train_dataset = dataset["train"].with_transform(preprocess_train)
    val_dataset = dataset["val"].with_transform(preprocess_train)
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    return train_dataloader, val_dataloader


def main():
    args = parse_args()
    print(args)
    train_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run = wandb.init(
        # Set the project where this run will be logged
        project="sg_encoder",
        # Track hyperparameters and run metadata
        config=vars(args))

    model = build_model(args, device=device)
    model.to(device)

    loss_img = nn.CrossEntropyLoss()
    loss_sg = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-6,
                           weight_decay=0.2)

    train_dataloader, val_dataloader = build_dataloader(args, device=device)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=len(train_dataloader) * args.num_train_epochs,
    )
    global_step = 0
    min_loss = 100000
    for epoch in range(args.num_train_epochs):
        train_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            # imgs = batch['image']
            # boxes = batch[BOXES]
            triplets = batch[TRIPLETS]
            objects = batch[OBJECTS]
            latent = batch[IMAGE_LATENT]

            logit_img, logit_sg = model(triplets, objects, latent=latent)
            ground_truth = torch.arange(len(latent), dtype=torch.long, device=device)

            # calculate box loss with MSE
            # box_loss = torch.nn.MSELoss()
            # box_loss = box_loss(box_pred, boxes)

            total_loss = (loss_img(logit_img, ground_truth) + loss_sg(logit_sg, ground_truth)) / 2
            total_loss.backward()

            run.log({"step_loss": total_loss.item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
            optimizer.step()
            lr_scheduler.step()
            global_step += 1
            train_loss += total_loss.item()

        run.log({"train_loss": train_loss / len(train_dataloader)}, step=global_step)

        if epoch % args.validation_epochs == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for step, batch in enumerate(tqdm(val_dataloader)):
                    # imgs = batch['image']
                    # boxes = batch[BOXES]
                    triplets = batch[TRIPLETS]
                    objects = batch[OBJECTS]
                    latent = batch[IMAGE_LATENT]

                    logit_img, logit_sg = model(triplets, objects, latent=latent)
                    ground_truth = torch.arange(len(latent), dtype=torch.long, device=device)

                    total_loss = (loss_img(logit_img, ground_truth) + loss_sg(logit_sg, ground_truth)) / 2
                    val_loss += total_loss.item()

                run.log({"val_loss": val_loss / len(val_dataloader)}, step=global_step)
                if val_loss / len(val_dataloader) < min_loss:
                    print(f'Min loss improved from {min_loss} to {val_loss / len(val_dataloader)}')
                    min_loss = val_loss / len(val_dataloader)
                    torch.save(model.state_dict(), f'./model/sg_encoder_best_{train_id}.pt')

            model.train()

        if epoch % 10 == 0:
            # save the file with date identifier
            torch.save(model.state_dict(), f'./model/sg_encoder_{epoch}_{train_id}.pt')


if __name__ == '__main__':
    os.makedirs('./model', exist_ok=True)
    main()
