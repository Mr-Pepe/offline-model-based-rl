for seed in 0 1 2
do
python main.py \
    --env HalfCheetah-v2 \
    --seed $seed \
    --epochs 200 \
    --pretrain_epochs 0 \
    --steps_per_epoch 1000 \
    --init_steps 2000 \
    --random_steps 10000 \
    --buffer_size 1000000 \
    --hid 200 \
    --l 4 \
    --gamma 0.99 \
    --agent_updates_per_step 40 \
    --agent_batch_size 256 \
    --agent_lr 3e-4 \
    --use_model True \
    --model_type probabilistic \
    --n_networks 7 \
    --model_batch_size 256 \
    --model_lr 1e-3 \
    --model_val_split 0.2 \
    --model_patience 20 \
    --rollouts_per_step 400 \
    --rollout_schedule 1 1 20 100 \
    --train_model_every 250 \
    --num_test_episodes 10 \
    --exp_name half_cheetah_mbpo_online \
    --datestamp False \
    --log_dir "" \
    --device cuda \
    --render False
done
exit 0