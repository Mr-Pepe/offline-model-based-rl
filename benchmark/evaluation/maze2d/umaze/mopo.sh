for seed in 0
do
python main.py \
    --env maze2d-umaze-v1 \
    --seed $seed \
    --epochs 20 \
    --pretrain_epochs 100 \
    --steps_per_epoch 1000 \
    --init_steps 0 \
    --random_steps 3000 \
    --reset_masze2d_umaze True \
    --n_samples_from_dataset 10000 \
    --hid 200 \
    --l 4 \
    --gamma 0.99 \
    --agent_updates_per_step 1 \
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
    --rollout_schedule 1 1 1 200 \
    --continuous_rollouts True \
    --max_rollout_length 500 \
    --train_model_every 1000 \
    --model_max_n_train_batches 1300 \
    --model_pessimism 5 \
    --exploration_mode state \
    --reset_buffer False \
    --num_test_episodes 3 \
    --exp_name maze2d_umaze_mopo \
    --datestamp False \
    --log_dir "" \
    --device cuda \
    --render False
done
exit 0
