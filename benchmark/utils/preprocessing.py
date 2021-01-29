from benchmark.utils.envs import HALF_CHEETAH_EXPERT, HALF_CHEETAH_MEDIUM, HALF_CHEETAH_MEDIUM_EXPERT, HALF_CHEETAH_MEDIUM_REPLAY, HALF_CHEETAH_RANDOM, HOPPER_EXPERT, HOPPER_MEDIUM, HOPPER_MEDIUM_EXPERT, HOPPER_MEDIUM_REPLAY, HOPPER_RANDOM, WALKER_EXPERT, WALKER_MEDIUM, WALKER_MEDIUM_EXPERT, WALKER_MEDIUM_REPLAY, WALKER_RANDOM
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


def preprocess_half_cheetah_expert(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([-0.0451,  0.1598,  0.0303, -0.1751, -0.2191, -0.0780,  0.0971, -0.0416,
                            10.0858, -0.0790, -0.2494,  0.3359,  0.4061,  0.3639,  0.7805, -0.3103,
                            -0.0542, -0.0440, -0.2340, -0.4979, -0.1429,  0.0148, -0.1501],
                           device=obs_act.device)

    std = torch.as_tensor([0.0695,  1.0903,  0.5122,  0.4170,  0.2232,  0.6120,  0.2733,  0.2680,
                           2.4657,  0.6230,  1.7847, 11.1089, 12.0512,  6.9271, 13.1120,  6.7304,
                           5.9150,  0.8653,  0.7456,  0.6908,  0.7162,  0.7298,  0.6550],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_half_cheetah_random(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([-1.5909e-01,  7.0300e-01,  3.0358e-03,  1.8578e-02, -3.0842e-03,
                            3.5867e-02, -1.9903e-02, -2.0291e-02, -8.9374e-02, -1.3927e-02,
                            1.2242e-02,  2.1843e-02, -6.1578e-02,  3.2814e-02,  5.9915e-03,
                            3.8224e-02,  6.3421e-03, -6.7090e-04, -3.3889e-04,  2.1943e-05,
                            8.2727e-05,  5.1789e-04,  3.2931e-04],
                           device=obs_act.device)

    std = torch.as_tensor([0.1893, 1.2229, 0.2578, 0.2750, 0.2662, 0.3321, 0.3079, 0.2851, 0.7243,
                           0.7381, 1.5444, 5.6747, 6.8373, 7.0526, 6.4119, 6.7524, 6.0904, 0.5776,
                           0.5775, 0.5776, 0.5775, 0.5777, 0.5778],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_half_cheetah_medium_expert(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([-0.0578,  0.0995, -0.0760, -0.2251, -0.2788, -0.0849, -0.0576, -0.0643,
                            7.6171, -0.0607, -0.1426,  0.2383,  0.2323,  0.2279,  0.4244, -0.1533,
                            -0.0206, -0.1821, -0.3215, -0.6153, -0.1321, -0.2260, -0.1534],
                           device=obs_act.device)

    std = torch.as_tensor([0.0765,  0.8204,  0.4338,  0.3858,  0.2113,  0.5621,  0.3071,  0.3013,
                           3.1609,  0.6945,  1.8874,  9.1251, 10.0269,  5.8362, 11.9091,  6.2311,
                           6.7531,  0.8472,  0.7145,  0.6362,  0.7001,  0.7278,  0.6879],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_hopper_random(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([1.2273e+00, -6.1983e-02, -4.3004e-02, -6.2249e-02,  4.3225e-02,
                            -1.5939e-01, -2.4343e-01, -1.1214e+00, -7.7519e-01, -8.6454e-01,
                            4.5092e-01,  4.1513e-05, -1.0491e-04, -2.0277e-04],
                           device=obs_act.device)

    std = torch.as_tensor([0.0247, 0.0633, 0.0643, 0.1024, 0.1509, 0.4084, 0.2987, 1.3903, 1.4430,
                           1.4840, 1.7350, 0.5776, 0.5773, 0.5775],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_hopper_medium(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([1.0285e+00, -1.0670e-01, -1.5392e-03, -1.1538e+00,  7.2936e-01,
                            5.0019e-01, -4.3483e-01, -1.2107e-02, -7.5291e-03, -1.7051e+00,
                            5.1350e-01,  6.0635e-01,  9.3471e-03,  6.6806e-01],
                           device=obs_act.device)

    std = torch.as_tensor([0.1600, 0.0327, 0.0105, 0.5526, 0.1624, 0.2380, 0.5784, 0.4661, 0.3273,
                           0.7685, 2.4598, 0.4637, 0.3298, 0.4812],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_hopper_medium_replay(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([1.2451,  0.0231, -0.2818, -0.3319,  0.1226,  2.0493, -0.1399,  0.0193,
                            -0.2790, -0.2019,  0.0021,  0.0442,  0.0486, -0.1263],
                           device=obs_act.device)

    std = torch.as_tensor([0.1592, 0.0696, 0.2358, 0.2834, 0.5703, 1.0523, 1.2749, 0.9437, 2.0762,
                           2.9279, 5.5717, 0.4752, 0.5847, 0.6789],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_hopper_medium_expert(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([1.1419, -0.0446, -0.1921, -0.7269,  0.3994,  1.4767, -0.2804, -0.0044,
                            -0.1252, -0.9139,  0.2974,  0.3045,  0.0276,  0.2545],
                           device=obs_act.device)

    std = torch.as_tensor([0.1929, 0.0794, 0.2406, 0.6063, 0.5459, 1.1919, 1.0353, 0.7059, 1.5226,
                           2.4569, 4.7061, 0.5426, 0.4862, 0.7200],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_hopper_expert(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([1.2560,  0.0175, -0.3803, -0.2997,  0.0681,  2.4628, -0.1259,  0.0033,
                            -0.2430, -0.1225,  0.0812,  0.0026,  0.0458, -0.1586],
                           device=obs_act.device)

    std = torch.as_tensor([0.1529, 0.0620, 0.2097, 0.2550, 0.5937, 0.9267, 1.3272, 0.8827, 2.1219,
                           3.1985, 6.1769, 0.4380, 0.6028, 0.6817],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_walker_random(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([1.2005e+00, -3.5051e-01, -2.1683e-01, -3.0372e-01,  2.5575e-01,
                            -2.1869e-01, -2.9801e-01,  2.5205e-01, -8.8394e-01, -9.4614e-01,
                            -4.7904e+00, -3.0537e+00, -3.1154e+00,  1.2333e+00, -3.0596e+00,
                            -3.0795e+00,  1.2012e+00,  1.1066e-04,  4.2039e-05, -1.1647e-03,
                            -2.2870e-04,  3.9766e-04, -4.0958e-05],
                           device=obs_act.device)

    std = torch.as_tensor([0.0593, 0.3028, 0.2785, 0.4247, 0.4046, 0.2790, 0.4186, 0.4084, 0.5838,
                           0.9833, 4.7057, 5.9832, 5.3067, 7.4124, 5.9868, 5.3072, 7.4135, 0.5777,
                           0.5778, 0.5773, 0.5780, 0.5772, 0.5775],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_walker_medium(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([1.2242,  0.1637, -0.0389, -0.1410,  0.5397, -0.0314, -0.4620,  0.0300,
                            2.4538, -0.0333,  0.0511, -0.0246, -0.1057,  0.0936, -0.0042, -0.1246,
                            -0.5578,  0.4630,  0.3515,  0.4456,  0.7674,  0.0298, -0.1330],
                           device=obs_act.device)

    std = torch.as_tensor([0.1174, 0.3012, 0.1155, 0.2635, 0.5393, 0.1929, 0.3729, 0.7390, 1.2013,
                           0.8049, 1.5591, 1.8162, 3.0536, 4.0526, 1.4324, 3.7664, 5.6389, 0.5070,
                           0.5176, 0.6980, 0.3528, 0.5328, 0.7405],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_walker_expert(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([1.2384e+00,  1.9592e-01, -1.0488e-01, -1.8577e-01,  2.3020e-01,
                            2.2739e-02, -3.7382e-01,  3.3736e-01,  3.9236e+00, -4.7796e-03,
                            2.5380e-02, -5.6899e-03, -1.8003e-02, -4.7798e-01,  6.6257e-04,
                            -8.1859e-04,  6.8521e-03,  4.3935e-01,  2.9337e-01,  2.7342e-01,
                            8.0970e-01,  1.0854e-01,  2.5088e-01],
                           device=obs_act.device)

    std = torch.as_tensor([0.0668, 0.1704, 0.1736, 0.2187, 0.7462, 0.0244, 0.3730, 0.6229, 0.9724,
                           0.7302, 1.5059, 2.4981, 3.5159, 5.3649, 0.7978, 4.3177, 6.1804, 0.5192,
                           0.5194, 0.6883, 0.2659, 0.5347, 0.7243],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_walker_medium_replay(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([1.2094,  0.1326, -0.1437, -0.2047,  0.5578, -0.0323, -0.2785,  0.1913,
                            1.4700, -0.1251,  0.0564, -0.1001, -0.3400,  0.0353, -0.0893, -0.2996,
                            -0.5981,  0.2774,  0.3647,  0.4011,  0.6208,  0.2002,  0.0159],
                           device=obs_act.device)

    std = torch.as_tensor([0.1193, 0.3563, 0.2585, 0.4208, 0.5202, 0.1569, 0.3677, 0.7161, 1.3764,
                           0.8633, 2.6369, 3.0145, 3.7218, 4.8678, 2.6692, 3.8458, 5.4765, 0.5431,
                           0.5890, 0.6473, 0.4548, 0.5756, 0.7632],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_walker_medium_expert(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([1.2321e+00,  1.7979e-01, -7.1895e-02, -1.6340e-01,  3.8396e-01,
                            -4.3268e-03, -4.1909e-01,  1.8357e-01,  3.1913e+00, -1.9048e-02,
                            3.8232e-02, -1.5101e-02, -6.1875e-02, -1.9218e-01, -1.7649e-03,
                            -6.2679e-02, -2.7553e-01,  4.5139e-01,  3.2250e-01,  3.5996e-01,
                            7.9039e-01,  6.9182e-02,  5.8900e-02],
                           device=obs_act.device)

    std = torch.as_tensor([0.0958, 0.2452, 0.1511, 0.2432, 0.6693, 0.1401, 0.3756, 0.7005, 1.3170,
                           0.7686, 1.5327, 2.1838, 3.2931, 4.7627, 1.1594, 4.0518, 5.9225, 0.5133,
                           0.5193, 0.6985, 0.3131, 0.5352, 0.7572],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


preprocessing_functions = {
    HALF_CHEETAH_MEDIUM_REPLAY: preprocess_half_cheetah_medium_replay,
    HALF_CHEETAH_MEDIUM: preprocess_half_cheetah_medium,
    HALF_CHEETAH_EXPERT: preprocess_half_cheetah_expert,
    HALF_CHEETAH_RANDOM: preprocess_half_cheetah_random,
    HALF_CHEETAH_MEDIUM_EXPERT: preprocess_half_cheetah_medium_expert,
    HOPPER_RANDOM: preprocess_hopper_random,
    HOPPER_MEDIUM: preprocess_hopper_medium,
    HOPPER_MEDIUM_REPLAY: preprocess_hopper_medium_replay,
    HOPPER_MEDIUM_EXPERT: preprocess_hopper_medium_expert,
    HOPPER_EXPERT: preprocess_hopper_expert,
    WALKER_RANDOM: preprocess_walker_random,
    WALKER_MEDIUM: preprocess_walker_medium,
    WALKER_EXPERT: preprocess_walker_expert,
    WALKER_MEDIUM_REPLAY: preprocess_walker_medium_replay,
    WALKER_MEDIUM_EXPERT: preprocess_walker_medium_expert,
    'antmaze-umaze-v0': preprocess_antmaze_umaze,
    'antmaze-medium-diverse-v0': preprocess_antmaze_medium_diverse
}


def get_preprocessing_function(env_name):
    if env_name in preprocessing_functions:
        return preprocessing_functions[env_name]

    print("No preprocessing function found for {}".format(env_name))
    return None
