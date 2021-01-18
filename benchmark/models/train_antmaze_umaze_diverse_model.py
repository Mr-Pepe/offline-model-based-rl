import argparse
from benchmark.utils.postprocessing import get_postprocessing_function
from benchmark.utils.preprocessing import get_preprocessing_function
from benchmark.models.environment_model import EnvironmentModel
from benchmark.utils.load_dataset import load_dataset_from_env
import gym
import d4rl  # noqa
import torch
from benchmark.utils.str2bool import str2bool

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--augment', type=str2bool, default=False)
    parser.add_argument('--rew_amp', type=int, default=100)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make('antmaze-umaze-diverse-v0')

    if args.augment:
        print("Load augmented dataset")
        buffer = torch.load(
            '/home/felipe/Projects/thesis-code/data/datasets/antmaze_umaze_diverse_augmented.p')
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        save_path = '/home/felipe/Projects/thesis-code/data/models/antmaze_umaze_diverse/augmented_model.pt'
    else:
        print("Load non-augmented dataset")
        buffer, obs_dim, act_dim = load_dataset_from_env(env,
                                                         buffer_device=device)
        save_path = '/home/felipe/Projects/thesis-code/data/models/antmaze_umaze_diverse/model.pt'

    buffer.rew_buf *= args.rew_amp
    print("Amplified reward with factor {}".format(args.rew_amp))

    model = EnvironmentModel(obs_dim=obs_dim,
                             act_dim=act_dim,
                             hidden=4*[512],
                             type='probabilistic',
                             n_networks=15,
                             device=device,
                             pre_fn=get_preprocessing_function(
                                 'antmaze-umaze-diverse-v0'),
                             post_fn=get_postprocessing_function(
                                 'antmaze-umaze-diverse-v0'),
                             )

    model.train_to_convergence(buffer, debug=True, batch_size=128, lr=5e-4,
                               max_n_train_batches=-1, patience=15,
                               val_split=0.1)
    model.cpu()

    torch.save(model, save_path)
