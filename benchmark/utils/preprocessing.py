import torch


def preprocess_maze2d_umaze(obs_act):
    mean = torch.as_tensor([1.8653e+00,  2.0270e+00, -5.9192e-02, -6.7519e-02,
                            5.5773e-04, -2.4673e-04], device=obs_act.device)

    std = torch.as_tensor([0.9751, 0.9118, 0.8677, 0.9782, 0.5770, 0.5776],
                          device=obs_act.device)

    obs_act -= mean
    obs_act /= std

    return obs_act
