import argparse
import os
from pathlib import Path
from typing import Optional

import ray
import torch
from ray import tune
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from offline_mbrl.models.environment_model import EnvironmentModel
from offline_mbrl.user_config import MODELS_DIR
from offline_mbrl.utils.load_dataset import load_dataset_from_env
from offline_mbrl.utils.preprocessing import get_preprocessing_function
from offline_mbrl.utils.replay_buffer import ReplayBuffer
from offline_mbrl.utils.termination_functions import get_termination_function


def training_function(
    model_config: dict,
    data: ReplayBuffer,
    model_save_path: Optional[Path] = None,
    tuning: bool = True,
) -> None:
    model = EnvironmentModel(
        hidden_layer_sizes=4 * model_config["n_hidden"], **model_config
    )

    model.train_to_convergence(
        replay_buffer=data, checkpoint_dir=None, tuning=tuning, **model_config
    )

    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model, model_save_path)
        print(f"Saved model to: {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--n_hidden", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.device != "":
        device = args.device

    buffer, obs_dim, act_dim = load_dataset_from_env(
        args.env_name, buffer_device=device
    )

    pre_fn = get_preprocessing_function(args.env_name, device)
    termination_function = get_termination_function(args.env_name)

    # None values must be filled for tuning and final training
    config = {
        "device": device,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "max_n_train_epochs": 50,
        "obs_bounds_trainable": True,
        "r_bounds_trainable": True,
        "patience": args.patience,
        "type": "probabilistic",
        "n_networks": 7,
        "pre_fn": pre_fn,
        "termination_function": termination_function,
        "debug": False,
        "n_hidden": args.n_hidden,
        "lr": None,
        "batch_size": None,
    }

    if args.level == 0:
        # Perform training with tuned hyperparameters and save model

        save_path = MODELS_DIR / (args.env_name + "-model.pt")

        config.update(max_n_train_epochs=-1, debug=True)

        config.update(lr=args.lr, batch_size=256)

        assert config["lr"] is not None
        assert config["batch_size"] is not None

        training_function(
            model_config=config, data=buffer, model_save_path=save_path, tuning=False
        )
    else:
        if args.level == 1:
            config.update(
                lr=tune.loguniform(1e-5, 1e-2),
                batch_size=tune.choice([256, 512, 1024, 2048]),
            )

        assert config["lr"] is not None
        assert config["batch_size"] is not None

        ray.init()
        scheduler = ASHAScheduler(
            metric="val_loss", mode="min", time_attr="time_since_restore", max_t=1000
        )

        search_alg = HyperOptSearch(metric="val_loss", mode="min")

        save_name = args.env_name

        if args.augment_loss:
            save_name += "-aug-loss"

        save_name += "-model-tuning-lvl-" + str(args.level)

        analysis = tune.run(
            tune.with_parameters(training_function, data=buffer),
            name=save_name,
            config=config,
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=200,
            resources_per_trial={"gpu": 0.5},
            fail_fast=True,
        )

        print("Best config: ", analysis.get_best_config(metric="val_loss", mode="min"))
