python main.py \
    --env hopper-random-v0 \
    --hid 200 \
    --l 4 \
    --gamma 0.99 \
    --seed 0 \
    --epochs 125 \
    --pretrain_epochs 125 \
    --steps_per_epoch 1000 \
    --init_steps 0 \
    --random_steps 0 \
    --agent_updates_per_step 1 \
    --agent_batch_size 256 \
    --agent_lr 3e-4 \
    --use_model False \
    --num_test_episodes 10 \
    --exp_name hopper_sac_offline_random \
    --datestamp False \
    --log_dir "" \
    --device cuda \
    --render False