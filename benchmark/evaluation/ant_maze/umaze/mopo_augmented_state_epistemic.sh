for seed in 1
do
python main.py \
    --env antmaze-umaze-v0 \
    --seed $seed \
    --epochs 500 \
    --pretrain_epochs 0 \
    --steps_per_epoch 4000 \
    --init_steps 5000 \
    --random_steps 10000 \
    --env_steps_per_step 0 \
    --real_buffer_size 1000000 \
    --virtual_buffer_size 1000000 \
    --n_samples_from_dataset -1 \
    --hid 200 \
    --l 4 \
    --pretrained_agent_path "" \
    --gamma 0.99 \
    --agent_updates_per_step 3 \
    --agent_batch_size 256 \
    --agent_lr 3e-4 \
    --use_model True \
    --pretrained_model_path "/home/felipe/Projects/thesis-code/data/models/antmaze_umaze/augmented_model.pt" \
    --rollouts_per_step 100 \
    --rollout_schedule 1 1 1 200 \
    --continuous_rollouts True \
    --max_rollout_length 10 \
    --train_model_every 0 \
    --model_pessimism 50 \
    --exploration_mode state \
    --uncertainty epistemic \
    --reset_buffer False \
    --train_model_from_scratch False \
    --virtual_pretrain_epochs 0 \
    --use_custom_reward False \
    --num_test_episodes 10 \
    --exp_name antmaze_umaze_mopo_augmented_state_epistemic \
    --datestamp False \
    --log_dir "" \
    --device cuda \
    --render False
done
exit 0
