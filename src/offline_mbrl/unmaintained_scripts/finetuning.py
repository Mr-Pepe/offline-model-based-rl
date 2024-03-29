import argparse
import os

import numpy as np
import ray
import torch
from ray import tune

from offline_mbrl.train import Trainer
from offline_mbrl.user_config import MODELS_DIR
from offline_mbrl.utils.modes import PARTITIONING_MODES, PENALTY_MODES
from offline_mbrl.utils.setup_logger_kwargs import setup_logger_kwargs
from offline_mbrl.utils.str2bool import str2bool


def training_function(config, tuning=True):
    trainer = Trainer(**config)
    return trainer.train(tuning=tuning, quiet=True)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.device != "":
        device = args.device

    pretrained_model_name = args.env_name + "-model.pt"

    config = dict(
        env_name=args.env_name,
        agent_kwargs=dict(type="sac"),
        max_rollout_length=args.rollout_length,
        model_pessimism=None,
        ood_threshold=None,
        rollouts_per_step=50,
        model_kwargs=dict(patience=5),
        seed=0,
        epochs=args.epochs,
        steps_per_epoch=1000,
        random_steps=0,
        init_steps=999,
        env_steps_per_step=1,
        n_samples_from_dataset=args.n_samples_from_dataset,
        agent_updates_per_step=args.agent_updates_per_step,
        num_test_episodes=20,
        max_ep_len=1000,
        use_model=True,
        pretrained_agent_path=args.pretrained_agent_path,
        pretrained_model_path=os.path.join(MODELS_DIR, pretrained_model_name),
        mode=args.mode,
        model_max_n_train_batches=1000,
        rollout_schedule=[1, 1, 20, 100],
        continuous_rollouts=True,
        train_model_every=1000,
        use_custom_reward=False,
        real_buffer_size=int(2e6),
        virtual_buffer_size=int(1e6),
        reset_buffer=False,
        virtual_pretrain_epochs=0,
        train_model_from_scratch=False,
        pretrain_epochs=0,
        logger_kwargs={},
        save_freq=1,
        device=device,
        render=False,
    )

    if args.mode in PARTITIONING_MODES:
        ood_threshold = tune.grid_search(
            np.linspace(args.bounds[0], args.bounds[1], num=args.n_trials).tolist()
        )

        config.update(model_pessimism=0, ood_threshold=ood_threshold)

    elif args.mode in PENALTY_MODES:
        model_pessimism = tune.grid_search(
            np.linspace(args.bounds[0], args.bounds[1], num=args.n_trials).tolist()
        )

        config.update(model_pessimism=model_pessimism, ood_threshold=0)

    if args.use_ray:
        ray.init()

        tune.run(
            tune.with_parameters(training_function),
            name="finetuning-"
            + args.env_name
            + "-"
            + config["mode"]
            + "_"
            + str(args.bounds[0])
            + "_"
            + str(args.bounds[1]),
            config=config,
            max_failures=2,
            resources_per_trial={"gpu": 0.5},
        )

    else:
        config.update(
            model_pessimism=0,
            ood_threshold=0,
            logger_kwargs=setup_logger_kwargs("test-finetuning", seed=0),
        )
        training_function(config, tuning=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--pretrained_agent_path", type=str, default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--bounds", default=[0, 1], nargs=2, type=float)
    parser.add_argument("--rollout_length", type=int, default=3)
    parser.add_argument("--agent_updates_per_step", type=int, default=1)
    parser.add_argument("--n_samples_from_dataset", type=int, default=50000)
    parser.add_argument("--use_ray", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default="")
    main(parser.parse_args())
