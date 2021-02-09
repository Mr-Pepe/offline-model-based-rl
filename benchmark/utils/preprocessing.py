from benchmark.utils.envs import HALF_CHEETAH_EXPERT, HALF_CHEETAH_EXPERT_V1, HALF_CHEETAH_MEDIUM, HALF_CHEETAH_MEDIUM_EXPERT, HALF_CHEETAH_MEDIUM_EXPERT_V1, HALF_CHEETAH_MEDIUM_REPLAY, HALF_CHEETAH_MEDIUM_REPLAY_V1, HALF_CHEETAH_MEDIUM_V1, HALF_CHEETAH_RANDOM, HALF_CHEETAH_RANDOM_V1, HOPPER_EXPERT, HOPPER_EXPERT_V1, HOPPER_MEDIUM, HOPPER_MEDIUM_EXPERT, HOPPER_MEDIUM_EXPERT_V1, HOPPER_MEDIUM_REPLAY, HOPPER_MEDIUM_REPLAY_V1, HOPPER_MEDIUM_V1, HOPPER_RANDOM, HOPPER_RANDOM_V1, WALKER_EXPERT, WALKER_EXPERT_v1, WALKER_MEDIUM, WALKER_MEDIUM_EXPERT, WALKER_MEDIUM_EXPERT_V1, WALKER_MEDIUM_REPLAY, WALKER_MEDIUM_REPLAY_V1, WALKER_MEDIUM_v1, WALKER_RANDOM, WALKER_RANDOM_v1
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

    mean = torch.as_tensor([-1.6007e-01,  8.2988e-01,  1.3947e-02,  8.8892e-02,  5.1835e-02,
                            2.8807e-03, -1.2516e-01, -1.7123e-01,  2.6785e+00, -4.8868e-02,
                            -5.2643e-02,  3.5605e-03, -1.6394e-01, -5.1860e-02, -2.2703e-03,
                            1.0839e-01,  7.7343e-02, -4.9057e-02,  1.5118e-01, -1.3211e-03,
                            -4.3150e-02, -2.1899e-01, -3.4953e-01],
                           device=obs_act.device)

    std = torch.as_tensor([0.2118,  1.4757,  0.4006,  0.3698,  0.4262,  0.4828,  0.3859,  0.3270,
                           2.2882,  0.9171,  1.6868, 10.4849,  9.2045, 11.4256,  7.9946,  8.6810,
                           6.4746,  0.8279,  0.7782,  0.8029,  0.7143,  0.7472,  0.7124],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_half_cheetah_medium(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([-0.1100,  0.1568,  0.1038,  0.1469,  0.0784, -0.2011, -0.0822, -0.2802,
                            4.4634, -0.0758, -0.0926,  0.4187, -0.4117,  0.1163, -0.0600, -0.0974,
                            -0.1454,  0.0638,  0.2187,  0.0457, -0.2871, -0.0832, -0.5439],
                           device=obs_act.device)

    std = torch.as_tensor([0.1086,  0.6115,  0.4914,  0.4487,  0.3972,  0.4815,  0.3060,  0.2638,
                           1.9019,  0.9390,  1.6246, 14.4366, 11.9981, 11.9917, 12.1620,  8.1277,
                           6.4206,  0.9280,  0.8516,  0.8949,  0.8152,  0.8859,  0.6110],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_half_cheetah_expert(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([-5.8326e-02,  1.4509e-01,  1.6762e-01, -1.2980e-01,  8.3118e-02,
                            -7.2668e-03, -2.3488e-02, -3.7352e-02,  1.2543e+01, -7.0531e-02,
                            -2.4624e-02,  1.8086e-01, -1.8211e-01, -3.0948e-01, -1.2119e-01,
                            1.8495e-01, -5.7062e-02,  2.1144e-01, -7.5213e-02, -1.0145e-02,
                            -7.1245e-02, -1.7979e-01, -8.3618e-02],
                           device=obs_act.device)

    std = torch.as_tensor([0.0741,  1.2578,  0.5837,  0.5008,  0.4536,  0.3896,  0.5706,  0.3779,
                           2.8547,  0.8444,  2.3568, 11.7626, 11.5059,  9.1782,  9.5548, 12.4762,
                           8.3821,  0.8812,  0.7119,  0.8867,  0.7096,  0.7682,  0.6798],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_half_cheetah_random(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([-2.0335e-01,  1.0081e+00,  7.3662e-04,  1.6249e-02,  8.5222e-03,
                            4.1635e-02,  1.5025e-03, -4.7313e-03, -7.1623e-02, -1.1384e-02,
                            2.1863e-02,  2.6099e-02, -7.4666e-02,  3.7844e-02,  4.4274e-03,
                            4.6928e-02,  9.2664e-03, -2.5432e-03,  8.1395e-03,  1.0160e-02,
                            1.1438e-04,  8.4745e-03,  6.1668e-03],
                           device=obs_act.device)

    std = torch.as_tensor([0.2261, 1.5071, 0.2778, 0.2968, 0.2857, 0.3563, 0.3293, 0.2952, 0.7608,
                           0.7699, 1.6083, 6.1483, 7.4135, 7.6633, 7.0019, 7.4380, 6.4921, 0.6243,
                           0.6241, 0.6260, 0.6249, 0.6232, 0.6248],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_half_cheetah_medium_expert(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([-0.0842,  0.1509,  0.1357,  0.0085,  0.0807, -0.1042, -0.0529, -0.1588,
                            8.5028, -0.0732, -0.0586,  0.2998, -0.2969, -0.0966, -0.0906,  0.0438,
                            -0.1012,  0.1376,  0.0718,  0.0178, -0.1792, -0.1315, -0.3138],
                           device=obs_act.device)

    std = torch.as_tensor([0.0965,  0.9889,  0.5405,  0.4952,  0.4264,  0.4485,  0.4588,  0.3478,
                           4.7119,  0.8930,  2.0244, 13.1682, 11.7551, 10.6802, 10.9364, 10.5298,
                           7.4662,  0.9079,  0.7985,  0.8912,  0.7718,  0.8305,  0.6860],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_hopper_random(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([1.0563,  0.0434, -0.6026, -0.0565,  0.3985,  1.6184, -0.6042,  0.0728,
                            -1.4327, -0.1136, -0.8023,  0.0198,  0.2046, -0.0766],
                           device=obs_act.device)

    std = torch.as_tensor([0.2329, 0.0390, 0.4602, 0.0854, 0.5309, 1.0275, 0.9329, 0.9123, 1.6886,
                           1.8751, 4.9047, 0.5135, 0.6110, 0.7400],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_hopper_medium(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([1.3002,  0.0151, -0.2698, -0.3320,  0.0439,  2.1277, -0.2084,  0.0049,
                            -0.4474, -0.1559, -0.3525,  0.1067,  0.1152, -0.1525],
                           device=obs_act.device)

    std = torch.as_tensor([0.1677, 0.0814, 0.2858, 0.2962, 0.6323, 0.8762, 1.4232, 0.9793, 1.8497,
                           3.1855, 5.6005, 0.5300, 0.5901, 0.6671],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_hopper_medium_replay(obs_act, reverse=False, only_std=False):
    if not reverse:
        obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([1.1503,  0.0089, -0.4596, -0.1805,  0.2150,  1.9202, -0.3828,  0.0643,
                            -0.8867, -0.0255, -0.0799,  0.0293,  0.2271, -0.0550],
                           device=obs_act.device)

    std = torch.as_tensor([0.1620, 0.0648, 0.3549, 0.1972, 0.6046, 1.0565, 1.1189, 0.9345, 1.5246,
                           1.9589, 4.8266, 0.4130, 0.5275, 0.6956],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    if reverse:
        obs_act *= std[:length]
        if not only_std:
            obs_act += mean[:length]
    else:
        obs_act -= mean[:length]
        obs_act /= std[:length]

    return obs_act


def preprocess_hopper_medium_expert(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([1.3449, -0.0382, -0.3909, -0.2107,  0.1478,  2.5069, -0.0658,  0.0054,
                            -0.1971, -0.0100,  0.0384,  0.0220,  0.1344,  0.0254],
                           device=obs_act.device)

    std = torch.as_tensor([0.1905, 0.0691, 0.2118, 0.2072, 0.6094, 0.7774, 1.4822, 1.0707, 1.6783,
                           2.4929, 5.7788, 0.4645, 0.6177, 0.7213],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_hopper_expert(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([1.3835e+00, -4.7635e-02, -3.7702e-01, -2.1669e-01,  1.3437e-01,
                            2.6250e+00, -2.0022e-03, -6.4167e-03, -5.8366e-02, -6.8434e-03,
                            6.2194e-02,  2.0583e-02,  1.1574e-01,  4.1584e-02],
                           device=obs_act.device)

    std = torch.as_tensor([0.1709, 0.0660, 0.1656, 0.2086, 0.6095, 0.6467, 1.5371, 1.0957, 1.6735,
                           2.5871, 5.9517, 0.4742, 0.6327, 0.7253],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_walker_random(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([1.2029e+00, -3.5071e-01, -2.1724e-01, -3.0310e-01,  2.4935e-01,
                            -2.1661e-01, -3.0244e-01,  2.4625e-01, -9.2649e-01, -9.5596e-01,
                            -4.8626e+00, -3.0685e+00, -3.1434e+00,  1.1397e+00, -3.0639e+00,
                            -3.1498e+00,  1.1127e+00,  1.1472e-03,  2.1448e-03, -3.2721e-03,
                            3.5960e-04, -1.9731e-03,  1.5665e-04],
                           device=obs_act.device)

    std = torch.as_tensor([0.0582, 0.3031, 0.2803, 0.4287, 0.4150, 0.2799, 0.4260, 0.4195, 0.6158,
                           1.0217, 4.8055, 6.1045, 5.4614, 7.5653, 6.1087, 5.4427, 7.5619, 0.6287,
                           0.6298, 0.6278, 0.6284, 0.6268, 0.6274],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_walker_medium(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([1.1666,  0.1836, -0.2578, -0.5593,  0.3340, -0.1588, -0.3359,  0.3263,
                            1.6481, -0.2386, -0.1795, -0.6965, -1.0953, -0.4000, -0.5029, -0.3354,
                            -0.4964,  0.1665,  0.2243,  0.2231,  0.4864,  0.2684,  0.0732],
                           device=obs_act.device)

    std = torch.as_tensor([0.1107, 0.4578, 0.4926, 0.7939, 0.6594, 0.3057, 0.4202, 0.6837, 1.5519,
                           1.0871, 4.3116, 4.6080, 5.2682, 6.4485, 4.5374, 4.8635, 5.9191, 0.6478,
                           0.6615, 0.7340, 0.5859, 0.5948, 0.7687],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_walker_expert(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([1.0590,  0.0613, -0.4804, -0.9892,  0.2857, -0.5005, -0.9018,  0.1510,
                            3.7265, -0.0673, -0.0931, -0.2748, -0.3010, -0.0783, -0.4372, -0.6675,
                            -0.2107,  0.3786,  0.3127,  0.1763,  0.5055,  0.1778,  0.0659],
                           device=obs_act.device)

    std = torch.as_tensor([0.1184, 0.3801, 0.6083, 0.9254, 0.7547, 0.6905, 0.8014, 0.7397, 1.4850,
                           1.1201, 5.2318, 5.6760, 6.9605, 6.3482, 5.9251, 7.0368, 6.1470, 0.6297,
                           0.6308, 0.7338, 0.6042, 0.6473, 0.7419],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_walker_medium_replay(obs_act, reverse=False, only_std=False):
    if not reverse:
        obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([1.1747,  0.0508, -0.2330, -0.0532,  0.5123, -0.1224, -0.2795,  0.2216,
                            0.7779, -0.2434, -0.0226, -0.3072, -0.4010,  0.0151, -0.3755, -0.3263,
                            -0.2357,  0.1701,  0.4682,  0.2845,  0.3219,  0.2164, -0.0066],
                           device=obs_act.device)

    std = torch.as_tensor([0.1027, 0.3922, 0.2977, 0.2163, 0.5210, 0.2255, 0.4338, 0.6734, 1.1535,
                           0.8086, 3.8521, 4.3947, 3.7194, 6.0661, 4.3106, 4.4877, 6.5245, 0.5718,
                           0.5520, 0.6719, 0.5947, 0.5590, 0.7040],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    if reverse:
        obs_act *= std[:length]
        if not only_std:
            obs_act += mean[:length]
    else:
        obs_act -= mean[:length]
        obs_act /= std[:length]

    return obs_act


def preprocess_walker_medium_expert(obs_act):
    obs_act = obs_act.detach().clone()

    mean = torch.as_tensor([1.1133,  0.1224, -0.3695, -0.7745,  0.3094, -0.3295, -0.6188,  0.2384,
                            2.6870, -0.1530, -0.1363, -0.4857, -0.6983, -0.2392, -0.4701, -0.5014,
                            -0.3536,  0.2726,  0.2686,  0.1997,  0.4964,  0.2231,  0.0696],
                           device=obs_act.device)

    std = torch.as_tensor([0.1266, 0.4252, 0.5646, 0.8886, 0.7091, 0.5605, 0.6996, 0.7176, 1.8403,
                           1.1070, 4.7939, 5.1738, 6.1852, 6.4006, 5.2770, 6.0505, 6.0357, 0.6475,
                           0.6478, 0.7342, 0.5952, 0.6232, 0.7555],
                          device=obs_act.device)

    # This allows to preprocess an observation without action
    length = obs_act.shape[-1]

    obs_act -= mean[:length]
    obs_act /= std[:length]

    return obs_act


def preprocess_half_cheetah_medium_replay_v1(obs_act):
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


def preprocess_half_cheetah_medium_v1(obs_act):
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


def preprocess_half_cheetah_expert_v1(obs_act):
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


def preprocess_half_cheetah_random_v1(obs_act):
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


def preprocess_half_cheetah_medium_expert_v1(obs_act):
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


def preprocess_hopper_random_v1(obs_act):
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


def preprocess_hopper_medium_v1(obs_act):
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


def preprocess_hopper_medium_replay_v1(obs_act):
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


def preprocess_hopper_medium_expert_v1(obs_act):
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


def preprocess_hopper_expert_v1(obs_act):
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


def preprocess_walker_random_v1(obs_act):
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


def preprocess_walker_medium_v1(obs_act):
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


def preprocess_walker_expert_v1(obs_act):
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


def preprocess_walker_medium_replay_v1(obs_act):
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


def preprocess_walker_medium_expert_v1(obs_act):
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
    HALF_CHEETAH_RANDOM: preprocess_half_cheetah_random,
    HALF_CHEETAH_MEDIUM: preprocess_half_cheetah_medium,
    HALF_CHEETAH_EXPERT: preprocess_half_cheetah_expert,
    HALF_CHEETAH_MEDIUM_REPLAY: preprocess_half_cheetah_medium_replay,
    HALF_CHEETAH_MEDIUM_EXPERT: preprocess_half_cheetah_medium_expert,
    HOPPER_RANDOM: preprocess_hopper_random,
    HOPPER_MEDIUM: preprocess_hopper_medium,
    HOPPER_EXPERT: preprocess_hopper_expert,
    HOPPER_MEDIUM_REPLAY: preprocess_hopper_medium_replay,
    HOPPER_MEDIUM_EXPERT: preprocess_hopper_medium_expert,
    WALKER_RANDOM: preprocess_walker_random,
    WALKER_MEDIUM: preprocess_walker_medium,
    WALKER_EXPERT: preprocess_walker_expert,
    WALKER_MEDIUM_REPLAY: preprocess_walker_medium_replay,
    WALKER_MEDIUM_EXPERT: preprocess_walker_medium_expert,
    HALF_CHEETAH_RANDOM_V1: preprocess_half_cheetah_random_v1,
    HALF_CHEETAH_MEDIUM_V1: preprocess_half_cheetah_medium_v1,
    HALF_CHEETAH_EXPERT_V1: preprocess_half_cheetah_expert_v1,
    HALF_CHEETAH_MEDIUM_REPLAY_V1: preprocess_half_cheetah_medium_replay_v1,
    HALF_CHEETAH_MEDIUM_EXPERT_V1: preprocess_half_cheetah_medium_expert_v1,
    HOPPER_RANDOM_V1: preprocess_hopper_random_v1,
    HOPPER_MEDIUM_V1: preprocess_hopper_medium_v1,
    HOPPER_EXPERT_V1: preprocess_hopper_expert_v1,
    HOPPER_MEDIUM_REPLAY_V1: preprocess_hopper_medium_replay_v1,
    HOPPER_MEDIUM_EXPERT_V1: preprocess_hopper_medium_expert_v1,
    WALKER_RANDOM_v1: preprocess_walker_random_v1,
    WALKER_MEDIUM_v1: preprocess_walker_medium_v1,
    WALKER_EXPERT_v1: preprocess_walker_expert_v1,
    WALKER_MEDIUM_REPLAY_V1: preprocess_walker_medium_replay_v1,
    WALKER_MEDIUM_EXPERT_V1: preprocess_walker_medium_expert_v1,
    'antmaze-umaze-v0': preprocess_antmaze_umaze,
    'antmaze-medium-diverse-v0': preprocess_antmaze_medium_diverse
}


def get_preprocessing_function(env_name):
    if env_name in preprocessing_functions:
        return preprocessing_functions[env_name]

    print("No preprocessing function found for {}".format(env_name))
    return None
