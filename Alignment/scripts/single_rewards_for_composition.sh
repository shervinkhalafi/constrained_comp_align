# i have tested the code with 128 batch size, i.e 4 gpus x 8 batch size x 4 gradient accumulation steps, however you can change the batch size 
# or batch size division as per your requirements
for epochs in 10
do
    for reward_fn in 'hps' 'aesthetic' 'imagereward' 'pickscore' 'mps'
    do
    accelerate launch main.py \
        --num_epochs=$epochs \
        --train_samples_per_epoch=128 \
        --num_eval_samples=4 \
        --eval_batch_size=4 \
        --num_validation_samples=0 \
        --train_gradient_accumulation_steps=16 \
        --backprop_strategy='gaussian' \
        --sample_num_steps=50 \
        --reward_fn=$reward_fn \
        --constraint_list="[('kl', 0.0, 0.1), ('${reward_fn}', 0.0 , 1.0)]"  \
        --train_prompt_fn='simple_animals' \
        --eval_prompt_fn='eval_simple_animals' \
        --train_batch_size=4 \
        --tracker_project_name="alignprop_baselines" \
        --log_with='wandb' \
        --constrained 'True' \
        --use_nupi 'False'  \
        --normalize_constraints 'False' \
        --num_validation_samples=0 \
        --project_dir="adapters/${reward_fn}/epochs_${epochs}" \
        --dual_learning_rate 0.0
    done
done