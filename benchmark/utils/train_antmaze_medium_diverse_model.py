from benchmark.utils.mazes import ANTMAZE_MEDIUM_MAX, ANTMAZE_MEDIUM_MIN
from benchmark.utils.augmentation import antmaze_augmentation
from benchmark.utils.replay_buffer import ReplayBuffer
import os
import argparse
from benchmark.utils.postprocessing import get_postprocessing_function, postprocess_antmaze_medium
from benchmark.utils.preprocessing import get_preprocessing_function
from benchmark.models.environment_model import EnvironmentModel
from benchmark.utils.load_dataset import load_dataset_from_env
import gym
import d4rl  # noqa
import torch
from benchmark.utils.str2bool import str2bool

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rew_amp', type=int, default=0.05)
    parser.add_argument('--augmented', type=str2bool, default=True)
    parser.add_argument('--data_path', type=str, default='')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make('antmaze-medium-diverse-v0')

    buffer, obs_dim, act_dim = load_dataset_from_env(env,
                                                     buffer_device=device)

    save_path = '/home/felipe/Projects/thesis-code/data/models/antmaze_medium_diverse/model.pt'

    buffer.rew_buf *= args.rew_amp
    print("Amplified reward with factor {}".format(args.rew_amp))

    augmentation_fn = None

    if args.augmented:
        train_buffer = ReplayBuffer(
            obs_dim, act_dim, buffer.size, device=device)
        train_idx = ~postprocess_antmaze_medium(
            next_obs=buffer.obs2_buf[:buffer.size].unsqueeze(0), ant_radius=0.8)['dones'].view(-1)
        train_idx = torch.logical_and(train_idx, (buffer.obs2_buf[:, 2] > 0.3))

        train_buffer.store_batch(buffer.obs_buf[train_idx],
                                 buffer.act_buf[train_idx],
                                 buffer.rew_buf[train_idx],
                                 buffer.obs2_buf[train_idx],
                                 buffer.done_buf[train_idx],)

        def augmentation_fn(x, y): return antmaze_augmentation(x,
                                                               y,
                                                               ANTMAZE_MEDIUM_MIN,
                                                               ANTMAZE_MEDIUM_MAX,
                                                               ANTMAZE_MEDIUM_MIN,
                                                               ANTMAZE_MEDIUM_MAX)

        buffer = train_buffer

        save_path = '/home/felipe/Projects/thesis-code/data/models/antmaze_medium_diverse/augmented_model.pt'

    model = EnvironmentModel(obs_dim=obs_dim,
                             act_dim=act_dim,
                             hidden=4*[512],
                             type='probabilistic',
                             n_networks=15,
                             device=device,
                             pre_fn=get_preprocessing_function(
                                 'antmaze-medium-diverse-v0'),
                             post_fn=get_postprocessing_function(
                                 'antmaze-medium-diverse-v0'),
                             )

    model.train_to_convergence(buffer, debug=True, batch_size=128,
                               max_n_train_batches=-1, patience=2,
                               val_split=0.05, lr_schedule=(1e-4, 1e-3),
                               no_reward=True, augmentation_fn=augmentation_fn)
    model.cpu()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(model, save_path)
