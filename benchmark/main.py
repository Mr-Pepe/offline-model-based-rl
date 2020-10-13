import argparse
from benchmark.utils.run_utils import setup_logger_kwargs
import torch
import gym
from benchmark.models.mlp_actor_critic import MLPActorCritic
from benchmark.sac.sac import sac

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'

    print("Training on {}".format(device))

    torch.set_num_threads(torch.get_num_threads())

    sac(lambda: gym.make(args.env), actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs, device=device)
