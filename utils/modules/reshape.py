from typing import Sequence
import torch

class Reshape(torch.nn.Module):
    def __init__(self, out_shape:Sequence[int]):
        super().__init__()
        self.out_shape = out_shape
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x.view([-1, *self.out_shape])