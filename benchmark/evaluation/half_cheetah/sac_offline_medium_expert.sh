for seed in 0 1 2
do
python main.py \
    --env halfcheetah-medium-expert-v0 \
    --seed $seed \
    --epochs 100 \
    --pretrain_epochs 200 \
    --steps_per_epoch 1000 \
    --init_steps 0 \
    --random_steps 0 \
    --buffer_size 1000000 \
    --hid 200 \
    --l 4 \
    --gamma 0.99 \
    --agent_updates_per_step 1 \
    --agent_batch_size 256 \
    --agent_lr 3e-4 \
    --use_model False \
    --num_test_episodes 10 \
    --exp_name half_cheetah_sac_offline_medium_expert \
    --datestamp False \
    --log_dir "" \
    --device cuda \
    --render False
done
exit 0