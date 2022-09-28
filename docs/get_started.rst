===========
Get started
===========

This library provides a simple but high-quality baseline for playing around with
model-free and model-based reinforcement learning approaches in both online and offline
settings.


Installation
============

Install `Mujoco <https://mujoco.org/>`_ version 2.1.0

Clone the repository from Github to play and install the library with::

    git clone https://github.com/Mr-Pepe/offline-model-based-rl
    cd offline-model-based-rl
    pip install -e .

Configure default directories in the ``user config.py`` file or by setting the
``OMBRL_DATA_DIR`` and ``OMBRL_MODELS_DIR`` environment variables.


Overview
============

This library allows training model-free (`SAC <https://arxiv.org/abs/1801.01290>`_)
and model-based (`MBPO <https://bair.berkeley.edu/blog/2019/12/12/mbpo/>`_) reinforcement
learning (RL) agents in online and offline settings.
Offline training leverages the `d4rl <https://github.com/Farama-Foundation/d4rl>`_ benchmark datasets.
Currently, only the halfcheetah, walker-2d, and hopper datasets are supported for offline training.

Offline model-based training uses the concepts of pessimism proposed in
`MOPO <https://arxiv.org/abs/2005.13239>`_ and `MOReL <https://arxiv.org/abs/2005.05951>`_.
Similar to MBPO, both approaches train an environment model on an offline dataset and
use that model to generate synthetic data for training a model-free RL agent. However,
both approaches employ uncertainty estimation techniques to detect areas where the model
prediction quality can not be guaranteed. The uncertainty estimate is either used to
penalize the predicted reward or to terminate trajectories when the uncertainty passes
a certain threshold. This library refers to the former as "penalty" (because of the
continuous reward penalty) and the latter as "partitioning" (because it partitions the
state-action space into certain and uncertain areas).

Different uncertainty estimation techniques can be used to determine the environment
model's uncertainty. This library implements the environment model as an ensemble of
probabilistic neural networks. The epistemic uncertainty can be estimated via the
ensemble, while the aleatoric uncertainty is estimated by each network in the ensemble.

This results in four different modes for offline model-based reinforcement learning:

    - Continuous reward penalization based on epistemic uncertainty estimation
    - Continuous reward penalization based on aleatoric uncertainty estimation
    - State-action space partitioning based on epistemic uncertainty estimation
    - State-action space partitioning based on aleatoric uncertainty estimation


Agent training
=================

The training loop is implemented in :py:mod:`offline_mbrl.train` and can be configured
via :py:class:`.TrainerConfiguration`. Check out the trainer configuration's documentation
to find out about all the configuration options you can use to customize the training.

Let's start with the most basic case of training an SAC agent online::

    python src/offline_mbrl/scripts/train_agent.py --env_name hopper-medium-replay-v2 --mode sac

Most values are taken from the default trainer configuration and you can adapt them to your needs.

You can observe the training process by running::

    tensorboard --logdir data/experiments/hopper-medium-replay-v2-sac/hopper-medium-replay-v2-sac-s0/

Open the displayed url in your browser.
Your agent should show some learning progress after 10-15 epochs.

Evaluate the agent performance by running::

    python src/offline_mbrl/scripts/evaluate_agent.py --env_name hopper-v2 --exp_path data/experiments/hopper-medium-replay-v2-sac/hopper-medium-replay-v2-sac-s0/


Model-based training
====================

You can visualize the training of an environment model consisting of an ensemble of probabilistic networks by running::

    python -m offline_mbrl.scripts.train_probabilistic_model_ensemble_on_toy_dataset

Make sure to install a graphical backend for matplotlib.


You can directly







Hyperparameter tuning
=====================




# Run experiments

> **Note** </br>
> I have recently been refactoring the code and it is probably not gonna run in its current state.



## Train the models

```
python models/pretraining/train_mujoco_models.py --env_name hopper-medium-v2 --patience 30
```

## Offline model-based RL without pessimism

```
python evaluation/train_mujoco_policies.py --mode mbpo --seeds 6 --env_name halfcheetah-random-v2
```

## Offline model-free RL without pessimism

```
python evaluation/train_mujoco_policies.py --mode sac --seeds 6 --env_name halfcheetah-random-v2
```

## Train pessimistic model-based method

```
python evaluation/train_mujoco_policies.py --env_name halfcheetah-medium-expert-v2 --tuned_params True --mode epistemic-penalty --epochs 100 --seeds 4 --start_seed 2
```

## Finetuning

```
python evaluation/train_mujoco_policies.py --env_name halfcheetah-medium-v2 --mode sac --epochs 100 --seeds 4 --env_steps_per_step 1 --n_samples_from_dataset 50000 --device cpu --pretrained_agent_path "..."
```


## Survival mode

```
python evaluation/train_mujoco_policies.py --env_name halfcheetah-medium-v2 --mode survival --epochs 100 --seeds 4 --ood_threshold 3.7
```

## Offline exploration

```
python evaluation/train_mujoco_policies.py --env_name halfcheetah-medium-v2 --mode offline-exploration-penalty --epochs 100 --seeds 4 --ood_threshold 3.7 --pessimism 1
```


## Pessimism sweep

```
python evaluation/pessimism_sweep.py --env_name hopper-expert-v2 --mode epistemic-penalty --epochs 10 --n_trials 50 --bounds 80.2 100
```
