import argparse
import os

from offline_mbrl.models.environment_model import EnvironmentModel
from offline_mbrl.schemas import EnvironmentModelConfiguration
from offline_mbrl.user_config import MODELS_DIR
from offline_mbrl.utils.load_dataset import load_dataset_from_env
from offline_mbrl.utils.preprocessing import get_preprocessing_function
from offline_mbrl.utils.termination_functions import get_termination_function

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = args.device

    buffer, obs_dim, act_dim = load_dataset_from_env(
        args.env_name, buffer_device=device
    )

    preprocessing_function = get_preprocessing_function(args.env_name, device)
    termination_function = get_termination_function(args.env_name)

    model_config = EnvironmentModelConfiguration(
        type="probabilistic",
        n_networks=7,
        training_patience=args.patience,
    )

    model = EnvironmentModel(
        obs_dim=obs_dim,
        act_dim=act_dim,
        config=model_config,
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_save_path = MODELS_DIR / (args.env_name + "-model.pt")

    model.train_to_convergence(
        replay_buffer=buffer,
        config=model_config,
        model_save_path=model_save_path,
        debug=True,
    )
