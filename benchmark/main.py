import argparse

import gym
import torch

from benchmark.train import train
from benchmark.utils.run_utils import setup_logger_kwargs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--init_steps', type=int, default=5000)
    parser.add_argument('--model_rollouts', type=int, default=10)
    parser.add_argument('--train_model_every', type=int, default=100)
    parser.add_argument('--model_batch_size', type=int, default=256)
    parser.add_argument('--model_lr', type=float, default=1e-3)
    parser.add_argument('--model_val_split', type=float, default=0.2)
    parser.add_argument('--agent_updates', type=int, default=40)
    parser.add_argument('--num_test_episodes', type=int, default=3)
    parser.add_argument('--exp_name', type=str, default='mbpo')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'

    print("Training on {}".format(device))

    torch.set_num_threads(torch.get_num_threads())

    train(lambda: gym.make(args.env),
          sac_kwargs=dict(hidden_sizes=[args.hid]*args.l, gamma=args.gamma),
          seed=args.seed, epochs=args.epochs,
          logger_kwargs=logger_kwargs, device=device, init_steps=args.init_steps,
          use_model=True, steps_per_epoch=args.steps_per_epoch, 
          agent_updates=args.agent_updates, model_rollouts=args.model_rollouts,
          train_model_every=args.train_model_every, model_lr=args.model_lr, 
          model_val_split=args.model_val_split, 
          model_batch_size=args.model_batch_size,
          num_test_episodes=args.num_test_episodes)
