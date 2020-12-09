import argparse
from benchmark.utils.str2bool import str2bool
import torch
import d4rl  # noqa

from benchmark.train import Trainer
from benchmark.utils.run_utils import setup_logger_kwargs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='halfcheetah-medium-replay-v0')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=125)
    parser.add_argument('--pretrain_epochs', type=int, default=125)
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--init_steps', type=int, default=3000)
    parser.add_argument('--random_steps', type=int, default=5000)
    parser.add_argument('--real_buffer_size', type=int, default=int(1e6))
    parser.add_argument('--virtual_buffer_size', type=int, default=int(1e6))
    parser.add_argument('--n_samples_from_dataset', type=int, default=50000)
    parser.add_argument('--reset_maze2d_umaze', type=str2bool, default=True)

    parser.add_argument('--hid', type=int, default=200)
    parser.add_argument('--l', type=int, default=4)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--agent_updates_per_step', type=int, default=1)
    parser.add_argument('--agent_batch_size', type=int, default=256)
    parser.add_argument('--agent_lr', type=float, default=3e-4)

    parser.add_argument('--use_model', type=str2bool, default=False)
    parser.add_argument('--model_type', type=str, default='deterministic')
    parser.add_argument('--n_networks', type=int, default=1)
    parser.add_argument('--model_batch_size', type=int, default=1024)
    parser.add_argument('--model_lr', type=float, default=1e-3)
    parser.add_argument('--model_val_split', type=float, default=0.2)
    parser.add_argument('--model_patience', nargs='+', type=int, default=[20])
    parser.add_argument('--rollouts_per_step', type=int, default=100)
    parser.add_argument('--rollout_schedule', nargs='+',
                        type=int, default=[1, 1, 20, 100])
    parser.add_argument('--continuous_rollouts', type=str2bool, default=True)
    parser.add_argument('--max_rollout_length', type=int, default=20)
    parser.add_argument('--train_model_every', type=int, default=1000)
    parser.add_argument('--model_max_n_train_batches', type=int, default=1300)
    parser.add_argument('--model_pessimism', type=int, default=-1)
    parser.add_argument('--exploration_mode', type=str, default='state')
    parser.add_argument('--reset_buffer', type=str2bool, default=True)
    parser.add_argument('--train_model_from_scratch', type=str2bool, default=True)

    parser.add_argument('--num_test_episodes', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='sac_offline')
    parser.add_argument('--datestamp', type=str2bool, default=False)
    parser.add_argument('--log_dir', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--render', type=str2bool, default=True)
    args = parser.parse_args()

    log_dir = None if args.log_dir == '' else args.log_dir

    logger_kwargs = setup_logger_kwargs(args.exp_name,
                                        seed=args.seed,
                                        data_dir=log_dir,
                                        datestamp=args.datestamp)

    device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() \
        else 'cpu'

    print("Training on {}".format(device))

    torch.set_num_threads(torch.get_num_threads())

    trainer = Trainer(args.env,
                      sac_kwargs=dict(hidden=[args.hid]*args.l,
                                      gamma=args.gamma,
                                      pi_lr=args.agent_lr,
                                      q_lr=args.agent_lr,
                                      batch_size=args.agent_batch_size),
                      model_kwargs=dict(type=args.model_type,
                                        n_networks=args.n_networks,
                                        hidden=[args.hid]*args.l,
                                        lr=args.model_lr,
                                        val_split=args.model_val_split,
                                        batch_size=args.model_batch_size,
                                        patience=args.model_patience),
                      seed=args.seed,
                      epochs=args.epochs,
                      pretrain_epochs=args.pretrain_epochs,
                      steps_per_epoch=args.steps_per_epoch,
                      init_steps=args.init_steps,
                      random_steps=args.random_steps,
                      logger_kwargs=logger_kwargs,
                      device=device,
                      real_buffer_size=args.real_buffer_size,
                      virtual_buffer_size=args.virtual_buffer_size,
                      use_model=args.use_model,
                      model_pessimism=args.model_pessimism,
                      exploration_mode=args.exploration_mode,
                      model_max_n_train_batches=args.model_max_n_train_batches,
                      agent_updates_per_step=args.agent_updates_per_step,
                      rollouts_per_step=args.rollouts_per_step,
                      rollout_schedule=args.rollout_schedule,
                      continuous_rollouts=args.continuous_rollouts,
                      train_model_every=args.train_model_every,
                      num_test_episodes=args.num_test_episodes,
                      n_samples_from_dataset=args.n_samples_from_dataset,
                      train_model_from_scratch=args.train_model_from_scratch,
                      max_rollout_length=args.max_rollout_length,
                      reset_buffer=args.reset_buffer,
                      reset_maze2d_umaze=args.reset_maze2d_umaze,
                      render=args.render)

    trainer.train()
