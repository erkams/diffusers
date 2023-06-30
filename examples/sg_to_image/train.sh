nvidia-smi
pip list
pip install wandb
export PATH="/home/sekererkam/.local/bin:$PATH"
wandb login 8632f2214a3d81fe44564d0e4c4d89fe629a9bc0
huggingface-cli login --token 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
pip install -U xformers
git clone --branch sg_to_image https://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@github.com/erkams/diffusers.git
cd diffusers
pip install -e .
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="/mnt/students/experiments/diffusion/sg_to_image_e2e"
export DATASET="erkam/clevr-full-v4"
export WANDB_API_KEY="8632f2214a3d81fe44564d0e4c4d89fe629a9bc0"
wandb login 8632f2214a3d81fe44564d0e4c4d89fe629a9bc0
cd examples/sg_to_image
pip install -r requirements.txt

accelerate launch --mixed_precision="fp16" train_sg_to_image_lora.py \
--pretrained_model_name_or_path=$MODEL_DIR \
--dataset_name=$DATASET --caption_column="objects_str" \
--boxes_column="boxes" --objects_column="objects" --triplets_column="triplets" \
--resolution=512 --train_batch_size=1 --gradient_accumulation_steps=4 \
--num_train_epochs=100 --checkpointing_steps=5000 --learning_rate=1e-04 \
--lr_scheduler="constant_with_warmup" --lr_warmup_steps=1000 --seed=42 \
--output_dir="sd-clevr-sg2im-objects_cap-e2e" --cond_place="attn" \
--num_validation_images=4 --report_to="wandb" \
--push_to_hub --shuffle_triplets \
--train_sg --vocab_json="/mnt/students/vocab.json" \
--caption_type="objects" --lora_rank=4 --center_crop --leading_metric='FID'

export MODEL_DIR="stabilityai/stable-diffusion-2"

accelerate launch --mixed_precision="no" train_sg_to_image_lora.py \
--pretrained_model_name_or_path=$MODEL_DIR \
--dataset_name=$DATASET --caption_column="objects_str" --image_column="image" \
--boxes_column="boxes" --objects_column="objects" --triplets_column="triplets" \
--resolution=512 --train_batch_size=1 --gradient_accumulation_steps=4 \
--num_train_epochs=100 --checkpointing_steps=5000 --learning_rate=1e-04 \
--lr_scheduler="constant_with_warmup" --lr_warmup_steps=3000 --seed=42 \
--output_dir="sd-clevr-sg2im-objects_cap-e2e" --cond_place="attn" \
--num_validation_images=4 --report_to="wandb" \
--push_to_hub --shuffle_triplets \
--train_sg --start_lora=1 --max_train_steps=50000 --vocab_json="/home/erkam/simsg/simsg/data/clevr_gen/MyClevr/target/vocab.json" \
--caption_type="objects" --lora_rank=4 --center_crop --leading_metric='FID'