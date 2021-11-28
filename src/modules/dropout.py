from typing import Union

import torch
from torch import nn as nn

from src.modules.base_generator import GeneratorAbstract
from src.utils.torch_utils import Activation


class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)


class DropoutGenerator(GeneratorAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        return self.in_channel

    def __call__(self, repeat: int = 1):
        p = self.args[0]
        return self._get_module(Dropout(p=p))

