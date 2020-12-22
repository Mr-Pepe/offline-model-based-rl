from benchmark.utils.postprocessing import get_postprocessing_function
from benchmark.utils.preprocessing import get_preprocessing_function
from benchmark.models.environment_model import EnvironmentModel
from benchmark.utils.load_dataset import load_dataset_from_env
import gym
import d4rl  # noqa
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
env = gym.make('antmaze-medium-diverse-v0')
buffer, obs_dim, act_dim = load_dataset_from_env(env,
                                                 buffer_device=device)

model = EnvironmentModel(obs_dim=obs_dim,
                         act_dim=act_dim,
                         hidden=4*[512],
                         type='probabilistic',
                         n_networks=15,
                         device=device,
                         pre_fn=get_preprocessing_function('antmaze-medium-diverse-v0'),
                         post_fn=get_postprocessing_function('antmaze-medium-diverse-v0'),
                         )

max_n_train_batches = -1

model.train_to_convergence(buffer, debug=True, batch_size=128, lr=5e-4,
                           max_n_train_batches=max_n_train_batches, patience=30,
                           val_split=0.1)
model.cpu()

torch.save(
    model,
    '/home/felipe/Projects/thesis-code/data/models/antmaze_medium/model.pt')
