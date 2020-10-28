# SAC with multiple agent updates per environment step to compare to the
# multiple updates every step of MBPO

python main.py \
    --env HalfCheetah-v2 \
    --hid 200 \
    --l 4 \
    --gamma 0.99 \
    --seed 0 \
    --epochs 400 \
    --steps_per_epoch 1000 \
    --init_steps 1000 \
    --use_model False \
    --agent_updates 40 \
    --num_test_episodes 10 \
    --exp_name half_cheetah_vanilla_sac_multi_update \
    --device cuda
