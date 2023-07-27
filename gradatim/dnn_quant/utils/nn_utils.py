from typing import Callable

import torch.nn as nn
from torch import Tensor


class Reshape(nn.Module):
    def __init__(self, shape: Callable[[Tensor], tuple]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        shape = self.shape(x)
        return x.view(shape)
