# Offline Model-Based Reinforcement Learning


[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=555555)](https://pycqa.github.io/isort/)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![Type checks: mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)


This library provides a simple but high-quality baseline for playing around with model-free and model-based reinforcement learning approaches in both online and offline settings.

The code in this repository is based on the code written as part of my master's thesis on uncertainty
estimation in offline model-based reinforcement learning. You can find the thesis in
[thesis.pdf](thesis.pdf). Please [cite](#citation) accordingly.


# Setup

Install [Mujoco](https://mujoco.org/).

Clone and install the library:

```
git clone git@github.com:Mr-Pepe/offline-model-based-rl.git
cd offline-model-based-rl
python3.9 -m venv venv
source venv/bin/activate
pip install -e .
```

Set default directories in the [user config](user_config.py).


# Run experiments

> **Note** </br>
> I have recently been refactoring the code and it is probably not gonna run in its current state.


All the following commands need to be run from the benchmark/benchmark folder.

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


# Contribute

Clone the repo and install the package and all required development dependencies

```
pip install -e .[dev]
```

After making changes to the code, make sure that static checks and unit tests pass by running `tox`.
Tox only runs unit tests that are marked as `fast` or `medium`.
For faster feedback from unit tests, run `pytest -m fast`.
Please run the slow tests if you have a GPU available by executing `pytest -m slow`.

# Citation

Feel free to use the code but please cite the usage as:

```
@misc{peter2021ombrl,
    title={Investigating Uncertainty Estimation Methods for Offline Reinforcement Learning},
    author={Felipe Peter and Elie Aljalbout},
    year={2021}
}
```

# TODOs
- TODO: Add code coverage badge (upload coverage results)
- TODO: Add test results badge (from workflow results)
- TODO: Add linting badge (from workflow results)
- TODO: Add license badge
- TODO: Remove mentions of "benchmark"
- TODO: Remove mentions of "felipe"