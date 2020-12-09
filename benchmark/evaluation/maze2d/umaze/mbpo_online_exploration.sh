for seed in 0
do
python main.py \
    --env maze2d-umaze-v1 \
    --seed $seed \
    --epochs 20 \
    --pretrain_epochs 0 \
    --steps_per_epoch 1000 \
    --init_steps 2000 \
    --random_steps 3000 \
    --reset_maze2d_umaze True \
    --hid 200 \
    --l 4 \
    --gamma 0.99 \
    --agent_updates_per_step 5 \
    --agent_batch_size 256 \
    --agent_lr 3e-4 \
    --use_model True \
    --model_type probabilistic \
    --n_networks 7 \
    --model_batch_size 256 \
    --model_lr 1e-3 \
    --model_val_split 0.2 \
    --model_patience 20 \
    --rollouts_per_step 100 \
    --rollout_schedule 2 2 1 200 \
    --continuous_rollouts True \
    --max_rollout_length 500 \
    --train_model_every 1000 \
    --model_max_n_train_batches 1300 \
    --model_pessimism -1 \
    --exploration_mode state \
    --reset_buffer True \
    --num_test_episodes 3 \
    --exp_name maze2d_umaze_mbpo_online_exploration \
    --datestamp False \
    --log_dir "" \
    --device cuda \
    --render False
done
exit 0
