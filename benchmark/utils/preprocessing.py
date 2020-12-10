from benchmark.utils.envs import ENV_CATEGORIES
import torch


def preprocess_maze2d_umaze(obs_act):
    mean = torch.as_tensor([1.8653e+00,  2.0270e+00, -5.9192e-02, -6.7519e-02,
                            5.5773e-04, -2.4673e-04], device=obs_act.device)

    std = torch.as_tensor([0.9751, 0.9118, 0.8677, 0.9782, 0.5770, 0.5776],
                          device=obs_act.device)

    obs_act -= mean
    obs_act /= std

    return obs_act


def preprocess_hopper(obs_act):
    return obs_act


def preprocess_half_cheetah(obs_act):
    return obs_act


def preprocess_walker2d(obs_act):
    return obs_act


def preprocess_antmaze_umaze(obs_act):
    return obs_act


preprocessing_functions = {
    'hopper': preprocess_hopper,
    'half_cheetah': preprocess_half_cheetah,
    'walker2d': preprocess_walker2d,
    'antmaze_umaze': preprocess_antmaze_umaze,
    'maze2d_umaze': preprocess_maze2d_umaze,
}


def get_preprocessing_function(env_name):
    for fn_name in ENV_CATEGORIES:
        if env_name in ENV_CATEGORIES[fn_name]:
            return preprocessing_functions[fn_name]

    return None
