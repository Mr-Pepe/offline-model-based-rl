# Vanilla SAC without any specialties

python main.py \
    --env Walker2d-v2 \
    --hid 200 \
    --l 4 \
    --gamma 0.99 \
    --seed 0 \
    --epochs 300 \
    --steps_per_epoch 1000 \
    --init_steps 1000 \
    --random_steps 10000 \
    --agent_batch_size 256 \
    --agent_lr 3e-4 \
    --use_model False \
    --agent_updates 1 \
    --num_test_episodes 10 \
    --exp_name walker2d_vanilla_sac \
    --device cuda
