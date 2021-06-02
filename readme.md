# Setup

Install Mujoco.

Install the environment from environment.yml (in the benchmark folder) using conda.

```
conda env create -f environment.yml
```

Install D4RL with pip from the D4RL folder.
Install orl-metrics with pip from the orl-metrics folder.
Install benchmark with pip from the benchmark folder.
```
pip install .
```

Set default directories in benchmark/benchmark/user_config.py.

# Run experiments

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

