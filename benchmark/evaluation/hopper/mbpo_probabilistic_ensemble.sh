# MBPO with a probabilistic ensemble as model

python main.py \
    --env Hopper-v2 \
    --term_fn hopper \
    --hid 200 \
    --l 4 \
    --gamma 0.99 \
    --seed 0 \
    --epochs 125 \
    --steps_per_epoch 1000 \
    --init_steps 1300 \
    --random_steps 10000 \
    --agent_batch_size 256 \
    --agent_lr 3e-4 \
    --use_model True \
    --model_type probabilistic \
    --n_networks 7 \
    --model_rollouts 400 \
    --rollout_schedule 1 15 20 100 \
    --train_model_every 250 \
    --model_batch_size 256 \
    --model_lr 1e-3 \
    --model_val_split 0.2 \
    --model_patience 20 \
    --agent_updates 20 \
    --num_test_episodes 10 \
    --exp_name hopper_mbpo_probabilistic_ensemble_use_term_fn \
    --device cuda
