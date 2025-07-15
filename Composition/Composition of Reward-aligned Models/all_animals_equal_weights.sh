
BATCH_SIZE=4
for promt in "cheetah" "snail" "hippopotamus" "crocodile" "lobster" "octopus"
do
    accelerate launch train_dreambooth_lora_new.py \
        --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
        --instance_data_dir="dog" \
        --load_model_dir="output_test/" \
        --adapters="/home/ubuntu/cdf/implementations/AlignProp/checkpoints/adapters/aesthetic/epochs_10, /home/ubuntu/cdf/implementations/AlignProp/checkpoints/adapters/pickscore/epochs_10, /home/ubuntu/cdf/implementations/AlignProp/checkpoints/adapters/hps/epochs_10, /home/ubuntu/cdf/implementations/AlignProp/checkpoints/adapters/imagereward/epochs_10" \
        --validation_rewards="aesthetic,hps,pickscore,imagereward, mps" \
        --output_dir="output_test" \
        --prompts="$promt" \
        --instance_prompt="$promt" \
        --resolution=512 \
        --train_batch_size=$BATCH_SIZE \
        --checkpointing_steps=200 \
        --learning_rate=0.1 \
        --lr_scheduler="cosine" \
        --lr_num_cycles=1 \
        --lr_warmup_steps=5 \
        --max_train_steps=2 \
        --validation_prompt="$promt" \
        --validation_epochs=1 \
        --primal_per_dual=20 \
        --min_timestep=1 --max_timestep=999 \
        --lr_dual=0.0 \
        --constrained=1 \
        --const_thresholds="0.0" \
        --num_validation_images=4 \
        --num_inference_steps_val=25 \
        --num_inference_steps_train=25 \
        --rank=4 \
        --gradient_accumulation_steps=2 \
        --seed="0" \
        --guidance_scale=5.0 \
        --classifier_class_names="$promt" \
        --project_name="reward_adapters" \
        --report_to="wandb" \
        --gen_mode="cfg" 
done