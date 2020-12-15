from benchmark.utils.envs import ENV_CATEGORIES
import torch


def preprocess_maze2d_umaze(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([1.8653e+00,  2.0270e+00, -5.9192e-02, -6.7519e-02,
                            5.5773e-04, -2.4673e-04], device=obs_act.device)

    std = torch.as_tensor([0.9751, 0.9118, 0.8677, 0.9782, 0.5770, 0.5776],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_antmaze_umaze(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([
        5.1114e+00,  3.2357e+00,  5.4443e-01,  8.2719e-01, -1.6899e-04,
        -5.4980e-04,  6.3538e-02, -4.3353e-03,  8.6055e-01, -2.9642e-03,
        -8.6039e-01,  1.7215e-03, -8.6124e-01,  3.6168e-03,  8.6043e-01,
        7.4084e-04, -1.0363e-03,  5.6322e-03,  1.0253e-04,  2.0915e-04,
        4.2020e-03,  9.1933e-04, -3.7982e-03, -4.7156e-04,  1.4381e-03,
        -1.4222e-03,  5.6210e-04, -1.3198e-04, -3.3764e-03,
        0.0024,  0.0019, -0.0029,  0.0007, -0.0024, -0.0014, -0.0012, -0.0009],
        device=obs_act.device)

    std = torch.as_tensor([
        2.9919, 3.5018, 0.0576, 0.2624, 0.0533, 0.0529, 0.4871, 0.3972, 0.2740,
        0.3973, 0.2745, 0.3988, 0.2740, 0.3980, 0.2739, 0.4040, 0.4024, 0.4286,
        0.7133, 0.7098, 0.7995, 1.8461, 1.6154, 1.8532, 1.6198, 1.8498, 1.6218,
        1.8604, 1.6244,
        0.5766, 0.5775, 0.5788, 0.5769, 0.5787, 0.5779, 0.5760, 0.5778],
        device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_hopper(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor(
        [0.4982,  0.0009, -0.0892, -0.1062,  0.0413,  0.6030, -0.0670, -0.1053,
         -0.1612, -0.1475,  0.0463,  0.0133,  0.0145, -0.0380],
        device=obs_act.device)

    std = torch.as_tensor(
        [0.6143, 0.0488, 0.1826, 0.2178, 0.3215, 1.1208, 0.7118, 0.7581, 1.2504,
         1.6932, 3.1109, 0.3190, 0.3701, 0.4193],
        device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_half_cheetah(obs_act):
    return obs_act.detach().clone()


def preprocess_walker2d(obs_act):
    return obs_act.detach().clone()


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
