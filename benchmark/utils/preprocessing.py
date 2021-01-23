from benchmark.utils.envs import HALF_CHEETAH_MEDIUM, HALF_CHEETAH_MEDIUM_REPLAY
import torch


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


def preprocess_antmaze_medium_diverse(obs_act):
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


def preprocess_half_cheetah_medium_replay(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([-0.1288,  0.3736, -0.1500, -0.2348, -0.2842, -0.1310, -0.2016, -0.0652,
                            3.4770, -0.0278, -0.0149,  0.0767,  0.0126,  0.0275,  0.0237,  0.0099,
                            -0.0159, -0.2636, -0.3636, -0.6473, -0.2020, -0.4300, -0.1152],
                           device=obs_act.device)

    std = torch.as_tensor([0.1702, 1.2842, 0.3344, 0.3673, 0.2609, 0.4784, 0.3182, 0.3355, 2.0931,
                           0.8037, 1.9044, 6.5731, 7.5727, 5.0700, 9.1051, 6.0855, 7.2533, 0.7994,
                           0.6885, 0.6088, 0.6628, 0.6553, 0.7183],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_half_cheetah_medium(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([-7.0458e-02,  3.9261e-02, -1.8216e-01, -2.7503e-01, -3.3857e-01,
                            -9.1859e-02, -2.1238e-01, -8.6914e-02,  5.1485e+00, -4.2427e-02,
                            -3.5751e-02,  1.4064e-01,  5.8565e-02,  9.1873e-02,  6.8410e-02,
                            3.7346e-03,  1.2949e-02, -3.2010e-01, -4.0871e-01, -7.3186e-01,
                            -1.2125e-01, -4.6688e-01, -1.5668e-01],
                           device=obs_act.device)

    std = torch.as_tensor([0.0811,  0.3874,  0.3026,  0.3447,  0.1796,  0.5071,  0.2569,  0.3296,
                           1.3096,  0.7588,  1.9791,  6.5656,  7.4688,  4.4834, 10.5581,  5.6839,
                           7.4979,  0.8052,  0.6707,  0.5524,  0.6835,  0.6410,  0.7192],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


preprocessing_functions = {
    HALF_CHEETAH_MEDIUM_REPLAY: preprocess_half_cheetah_medium_replay,
    HALF_CHEETAH_MEDIUM: preprocess_half_cheetah_medium,
    'antmaze-umaze-v0': preprocess_antmaze_umaze,
    'antmaze-medium-diverse-v0': preprocess_antmaze_medium_diverse
}


def get_preprocessing_function(env_name):
    if env_name in preprocessing_functions:
        return preprocessing_functions[env_name]

    print("No preprocessing function found for {}".format(env_name))
    return None
