from benchmark.utils.postprocessing import get_postprocessing_function
from benchmark.utils.preprocessing import get_preprocessing_function
from benchmark.models.environment_model import EnvironmentModel
from benchmark.utils.load_dataset import load_dataset_from_env
import gym
import d4rl  # noqa
import torch

use_augmented_dataset = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
env = gym.make('antmaze-umaze-v0')

if use_augmented_dataset:
    buffer = torch.load(
        '/home/felipe/Projects/thesis-code/data/datasets/antmaze_umaze_augmented.p')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    save_path = '/home/felipe/Projects/thesis-code/data/models/antmaze_umaze/augmented_model.pt'
else:
    buffer, obs_dim, act_dim = load_dataset_from_env(env,
                                                     buffer_device=device)
    save_path = '/home/felipe/Projects/thesis-code/data/models/antmaze_umaze/model.pt'

buffer.rew_buf *= 100

model = EnvironmentModel(obs_dim=obs_dim,
                         act_dim=act_dim,
                         hidden=4*[512],
                         type='probabilistic',
                         n_networks=15,
                         device=device,
                         pre_fn=get_preprocessing_function('antmaze-umaze-v0'),
                         post_fn=get_postprocessing_function(
                             'antmaze-umaze-v0'),
                         )

max_n_train_batches = -1

model.train_to_convergence(buffer, debug=True, batch_size=128, lr=5e-4,
                           max_n_train_batches=max_n_train_batches, patience=15,
                           val_split=0.1)
model.cpu()

torch.save(model, save_path)
