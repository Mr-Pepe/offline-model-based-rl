from benchmark.models.mlp import mlp
import torch.nn as nn

class EnvironmentModel(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=[128, 128]):
        super().__init__()

        self.layers = mlp([in_dim] + hidden + [out_dim], nn.ReLU)

    def forward(self, x):
        #TODO: Transform angular input to [sin(x), cos(x)]
        return self.layers(x)