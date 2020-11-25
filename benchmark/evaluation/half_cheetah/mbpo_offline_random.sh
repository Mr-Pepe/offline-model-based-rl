for seed in 0 1 2
do
python main.py \
    --env halfcheetah-random-v0 \
    --seed $seed \
    --epochs 100 \
    --pretrain_epochs 100 \
    --steps_per_epoch 1000 \
    --init_steps 0 \
    --random_steps 0 \
    --buffer_size 1000000 \
    --hid 200 \
    --l 4 \
    --gamma 0.99 \
    --agent_updates_per_step 10 \
    --agent_batch_size 256 \
    --agent_lr 3e-4 \
    --use_model True \
    --model_type probabilistic \
    --n_networks 7 \
    --model_batch_size 256 \
    --model_lr 1e-3 \
    --model_val_split 0.2 \
    --model_patience 50 20 \
    --rollouts_per_step 80 \
    --rollout_schedule 5 5 20 100 \
    --train_model_every 250 \
    --model_max_n_train_batches 2000 \
    --model_pessimism 0 \
    --num_test_episodes 10 \
    --exp_name half_cheetah_mbpo_offline_random \
    --datestamp False \
    --log_dir "" \
    --device cuda \
    --render False
done
exit 0
