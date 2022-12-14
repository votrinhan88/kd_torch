from typing import Sequence
import torch

class Concatenate(torch.nn.Module):
    def forward(self, tensors:Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tensors, dim=1)