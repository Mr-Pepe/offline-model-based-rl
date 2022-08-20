import torch.nn as nn

def mlp(sizes, activation, output_activation=nn.Identity):
    # Based on https://spinningup.openai.com

    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)
