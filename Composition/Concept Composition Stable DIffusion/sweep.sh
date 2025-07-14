#!/bin/bash
#composition of models

# for i in {1..100}; do
#   CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train.py \
#       --num_epochs=50 \
#       --num_inference_steps=10 \
#       --guidance_scale_sampling=12.0 \
#       --guidance_scale_kl=1.0 \
#       --lr_dual=0.050 \
#       --batch_size=2 \
#       --wandb_logging=1 \
#       --num_images=50 \
#       --wandb_project="PROJECT NAME 1" \
#       --seed=42 \
#       --running_average_window=5
# done


CUDA_VISIBLE_DEVICES=1 accelerate launch train.py \
    --num_epochs=50 \
    --num_inference_steps=10 \
    --guidance_scale_sampling=12.0 \
    --guidance_scale_kl=120.0 \
    --lr_dual=0.050 \
    --batch_size=2 \
    --wandb_logging=1 \
    --num_images=50 \
    --wandb_project="PROJECT NAME 1" \
    --seed=42 \
    --running_average_window=5
