import torch.nn as nn


class DeterministicNN(nn.Module):
    def __init__(self, in_dim, out_dim, layers=None):
        super(DeterministicNN, self).__init__()
        self._layer = nn.Linear(in_dim, out_dim)

    def forward(self, input_):
        output = self._layer(input_)
        return output

