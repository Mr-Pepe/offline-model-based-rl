from benchmark.utils.termination_functions import get_termination_function
from benchmark.utils.preprocessing import get_preprocessing_function
from benchmark.models.environment_model import EnvironmentModel
from benchmark.utils.load_dataset import load_dataset_from_env
import gym
import d4rl  # noqa
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
env = gym.make('antmaze-umaze-v0')
buffer, obs_dim, act_dim = load_dataset_from_env(env,
                                                 buffer_device=device)

model = EnvironmentModel(obs_dim=obs_dim,
                         act_dim=act_dim,
                         hidden=[200, 200, 200, 200],
                         type='probabilistic',
                         n_networks=7,
                         device=device,
                         pre_fn=get_preprocessing_function('antmaze-umaze-v0'),
                         term_fn=get_termination_function('antmaze-umaze-v0'),
                         )

max_n_train_batches = -1

model.train_to_convergence(buffer, debug=True, batch_size=128, lr=5e-4,
                           max_n_train_batches=max_n_train_batches, patience=20)
model.cpu()

torch.save(
    model,
    '/home/felipe/Projects/thesis-code/data/models/antmaze_umaze/model.pt')
