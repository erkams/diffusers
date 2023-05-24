git clone --branch sg_to_image https://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@github.com/erkams/diffusers.git
cd diffusers
pip install -e .
pip install -r /mnt/nfs-students/requirements.txt
cp -f /mnt/nfs-students/modeling_clip.py /usr/local/lib/python3.10/dist-packages/transformers/models/clip/modeling_clip.py
cd examples/sg_to_image

accelerate launch --mixed_precision="fp16" train_sg_to_image_lora.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2" \
--dataset_name="erkam/clevr-with-depth-full-v2" --caption_column="pos_prompt" \
--boxes_column="target_box" --objects_column="target_obj" --triplets_column="target_tri" \
--resolution=512 --train_batch_size=1 --gradient_accumulation_steps=4 \
--num_train_epochs=100 --checkpointing_steps=5000 --learning_rate=1e-04 \
--lr_scheduler="constant_with_warmup" --lr_warmup_steps=1000 --seed=42 \
--output_dir="sd-clevr-sg2im-nocap" --cond_place="attn" \
--num_validation_images=4 --report_to="wandb" \
--push_to_hub --shuffle_triplets \
--sg_model_path="/mnt/nfs-students/experiments/clevr_clip/spade_64_clevr_clip_model.pt" \
--caption_type='none' --lora_rank=128 --center_crop --leading_metric='FID'