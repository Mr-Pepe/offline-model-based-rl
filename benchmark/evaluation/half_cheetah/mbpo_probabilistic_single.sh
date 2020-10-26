# MBPO with a single probabilistic network as model

python main.py \
    --env HalfCheetah-v2 \
    --hid 200 \
    --l 4 \
    --gamma 0.99 \
    --seed 0 \
    --epochs 400 \
    --steps_per_epoch 1000 \
    --init_steps 1300 \
    --use_model True \
    --model_type probabilistic \
    --n_networks 1 \
    --model_rollouts 400 \
    --train_model_every 250 \
    --model_batch_size 256 \
    --model_lr 1e-3 \
    --model_val_split 0.2 \
    --agent_updates 40 \
    --num_test_episodes 10 \
    --exp_name mbpo_probabilistic_single \
    --device cuda
