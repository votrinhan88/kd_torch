import torch
from .metric import Metric

class Mean(Metric):
    def __init__(self):
        self.reset()

    def update(self, new_entry:torch.Tensor) -> torch.Tensor:
        self.step += 1
        self.accum_value += new_entry
        self.value = self.accum_value/self.step
        return self.value

    def reset(self):
        self.step:int = 0
        self.accum_value = torch.zeros(1)
        self.value = self.accum_value/self.step
