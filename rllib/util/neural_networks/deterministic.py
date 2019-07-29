import torch.nn as nn


class DeterministicNN(nn.Module):
    def __init__(self, in_dim, out_dim, layers=None):
        super().__init__()
        layers_ = []
        if layers is None:
            layers = []
        for layer in layers:
            layers_.append(nn.Linear(in_dim, layer))
            layers_.append(nn.ReLU())
            in_dim = layer

        self._layers = nn.Sequential(*layers_)
        self._head = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self._head(self._layers(x))
