nvidia-smi
pip list

pip install wandb
export PATH="/home/sekererkam/.local/bin:$PATH"
wandb login 8632f2214a3d81fe44564d0e4c4d89fe629a9bc0

pip install xformers

git clone --branch sg_to_image https://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@github.com/erkams/diffusers.git
cd diffusers
pip install -e .

export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="/mnt/nfs-students/experiments/diffusion/controlnet_l2i"
export DATASET="erkam/clevr-full-v4"
export WANDB_API_KEY="8632f2214a3d81fe44564d0e4c4d89fe629a9bc0"
export CACHE_DIR="/mnt/nfs-students/cache/huggingface"

wandb login 8632f2214a3d81fe44564d0e4c4d89fe629a9bc0

ln -s /usr/lib/wsl/lib/libcuda.so [path to your env here]/lib/libcuda.so

cd examples/controlnet
pip install -r requirements.txt
pip install matplotlib bitsandbytes scipy

accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=$DATASET \
 --cache_dir=$CACHE_DIR \
 --caption_column="objects_str" --conditioning_image_column="colored_layout" --image_column="image" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "/mnt/nfs-students/val0.png" "/mnt/nfs-students/val1.png" \
 --validation_prompt "cyan cylinder, yellow cube, green sphere, red sphere, blue cube, green cube" "cyan sphere, purple cube, gray sphere, cyan cylinder, red cube, blue cube" \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --enable_xformers_memory_efficient_attention \
 --hub_token="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" --hub_model_id="erkam/controlnet-clevr-l2i" \
 --report_to="wandb" \
 --set_grads_to_none \
 --tracker_project_name controlnet-clevr-layout

 ####### CLOUD $$$$$$$$


export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="~/experiments/diffusion/controlnet_l2i"
export DATASET="erkam/clevr-full-v4"
export WANDB_API_KEY="8632f2214a3d81fe44564d0e4c4d89fe629a9bc0"
export CACHE_DIR="~/cache/huggingface"

accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=$DATASET \
 --cache_dir=$CACHE_DIR \
 --caption_column="objects_str" --conditioning_image_column="colored_layout" --image_column="image" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./val0.png" "./val1.png" \
 --validation_prompt "cyan cylinder, yellow cube, green sphere, red sphere, blue cube, green cube" "cyan sphere, purple cube, gray sphere, cyan cylinder, red cube, blue cube" \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --enable_xformers_memory_efficient_attention \
 --hub_token="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" --hub_model_id="erkam/controlnet-clevr-l2i" \
 --report_to="wandb" \
 --set_grads_to_none \
 --tracker_project_name controlnet-clevr-layout