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
    parser.add_argument('--model_rollouts', type=int, default=400)
    parser.add_argument('--agent_updates', type=int, default=40)
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
          logger_kwargs=logger_kwargs, device=device, init_steps=10000,
          use_model=True, steps_per_epoch=args.steps_per_epoch, 
          agent_updates=args.agent_updates, model_rollouts=args.model_rollouts)
