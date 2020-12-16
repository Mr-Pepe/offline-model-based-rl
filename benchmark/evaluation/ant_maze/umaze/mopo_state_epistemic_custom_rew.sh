for seed in 1 2 3
do
python main.py \
    --env antmaze-umaze-v0 \
    --seed $seed \
    --epochs 1000 \
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
    --pretrained_model_path "/home/felipe/Projects/thesis-code/data/models/antmaze_umaze/model.pt" \
    --model_type probabilistic \
    --n_networks 7 \
    --model_batch_size 256 \
    --model_lr 1e-3 \
    --model_val_split 0.2 \
    --model_patience 20 \
    --rollouts_per_step 100 \
    --rollout_schedule 1 1 1 200 \
    --continuous_rollouts True \
    --max_rollout_length 5 \
    --train_model_every 0 \
    --model_max_n_train_batches 1300 \
    --model_pessimism 5 \
    --exploration_mode state \
    --uncertainty epistemic \
    --reset_buffer False \
    --train_model_from_scratch False \
    --virtual_pretrain_epochs 0 \
    --use_custom_reward True \
    --num_test_episodes 10 \
    --exp_name antmaze_umaze_mopo_state_epistemic_custom_rew \
    --datestamp False \
    --log_dir "" \
    --device cuda \
    --render False
done
exit 0
