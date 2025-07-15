# i have tested the code with 128 batch size, i.e 4 gpus x 8 batch size x 4 gradient accumulation steps, however you can change the batch size 
# or batch size division as per your requirements
epochs=20
constraint_level=0.0
WEIGHT=0.2
kl_coeff=0.1
constraint_level=-0.5
REWARD="mps"
accelerate launch --num_processes 1 --main_process_port 29501 main.py \
    --num_epochs=$epochs \
    --train_samples_per_epoch=128 \
    --num_eval_samples=16 \
    --eval_batch_size=4 \
    --num_validation_samples=0 \
    --train_gradient_accumulation_steps=16 \
    --backprop_strategy='gaussian' \
    --sample_num_steps=50 \
    --constraint_list="[('kl', 0.0, $kl_coeff), ('saturation_low', $constraint_level, $WEIGHT), ('local_contrast_low', $constraint_level, $WEIGHT), ('mps', $constraint_level, $WEIGHT)]" \
    --train_prompt_fn='eval_simple_animals' \
    --eval_prompt_fn='eval_simple_animals' \
    --train_batch_size=4 \
    --tracker_project_name="alignprop_baselines" \
    --log_with='wandb' \
    --constrained 'True' \
    --use_nupi 'True'  \
    --normalize_constraints 'True' \
    --num_validation_samples=0 \
    --project_dir="all_rewards/${REWARD}_${constraint_level}" \
    --dual_learning_rate 0.0 \
    --nupi_kappa_i 0.05 \
    --nupi_kappa_p 0.0 \
    --nupi_nu 0.1 \
    --use_cached_scale 'True' \
    --sample_num_steps 15
constraint_level=0.0
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29501 main.py \
    --num_epochs=$epochs \
    --train_samples_per_epoch=128 \
    --num_eval_samples=16 \
    --eval_batch_size=4 \
    --num_validation_samples=0 \
    --train_gradient_accumulation_steps=16 \
    --backprop_strategy='gaussian' \
    --sample_num_steps=50 \
    --constraint_list="[('kl', 0.0, $kl_coeff), ('saturation_low', $constraint_level, $WEIGHT), ('local_contrast_low', $constraint_level, $WEIGHT), ('mps', $constraint_level, $WEIGHT)]" \
    --train_prompt_fn='eval_simple_animals' \
    --eval_prompt_fn='eval_simple_animals' \
    --train_batch_size=4 \
    --tracker_project_name="alignprop_baselines" \
    --log_with='wandb' \
    --constrained 'True' \
    --use_nupi 'True'  \
    --normalize_constraints 'True' \
    --num_validation_samples=0 \
    --project_dir="all_rewards/${REWARD}_${constraint_level}" \
    --dual_learning_rate 0.0 \
    --nupi_kappa_i 0.0 \
    --nupi_kappa_p 0.0 \
    --nupi_nu 0.0 \
    --use_cached_scale 'True' \
    --sample_num_steps 15