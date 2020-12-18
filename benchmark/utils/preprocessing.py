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


def preprocess_antmaze_medium(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([
        1.2001e+01,  1.2906e+01,  4.8439e-01,  5.4019e-01,  1.9542e-02,
        3.1740e-02,  9.0824e-02, -2.7411e-02,  6.8376e-01,  1.0748e-01,
        -7.0496e-01, -4.8868e-02, -7.8523e-01,  1.4361e-02,  6.9033e-01,
        2.8084e-02,  3.3337e-02,  3.3885e-04,  2.2274e-03,  6.7202e-05,
        2.1963e-03, -2.6155e-04,  1.1806e-02,  1.7435e-04, -1.1992e-02,
        -3.7501e-03, -1.2402e-02,  4.5720e-03,  1.2490e-02, -7.9297e-03,
        -4.1525e-01, -4.1696e-02, -4.4633e-01,  1.6400e-01,  3.8667e-01,
        -9.3123e-02,  1.9887e-01],
        device=obs_act.device)

    std = torch.as_tensor([
        7.1495, 6.1200, 0.1536, 0.4086, 0.3447, 0.3440, 0.5430, 0.4392, 0.2634,
        0.4359, 0.2782, 0.4418, 0.3088, 0.4478, 0.2639, 0.6737, 0.6929, 0.6414,
        0.9810, 1.0182, 1.0752, 2.3651, 1.7416, 2.5640, 1.5952, 2.5276, 1.7366,
        2.3933, 1.6385, 0.8066, 0.6985, 0.8100, 0.6990, 0.8050, 0.7353, 0.8060,
        0.8004],
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
    'antmaze_medium': preprocess_antmaze_medium
}


def get_preprocessing_function(env_name):
    for fn_name in ENV_CATEGORIES:
        if env_name in ENV_CATEGORIES[fn_name]:
            return preprocessing_functions[fn_name]

    return None
