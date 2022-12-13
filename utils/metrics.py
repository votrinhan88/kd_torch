from abc import ABC, abstractmethod

import torch

class Metric(ABC):
    def __repr__(self):
        return f"{self.__class__.__name__}(latest={self.latest:.4g})"

    """Base class for metrics."""
    @abstractmethod
    def update(self):
        """Update metric from given inputs."""        
        pass

    @abstractmethod
    def reset(self):
        """Reset tracked parameters, typically used when moving to a new epoch."""        
        pass

    @property
    def latest(self):
        """Return the latest metric value in a pythonic format."""
        return self.value.item()

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

class CategoricalAccuracy(Metric):
    def __init__(self):
        self.reset()
    
    def update(self, label:torch.Tensor, prediction:torch.Tensor):
        # `prediction` can be probability or logits
        pred_label = prediction.argmax(dim=1)
        self.num_observed += label.shape[0]

        self.label = torch.cat(tensors=[self.label, label], dim=0)
        self.pred_label = torch.cat(tensors=[self.pred_label, pred_label], dim=0)

        self.value = (self.label == self.pred_label).sum()/self.num_observed
        return self.value

    def reset(self):
        self.label = torch.Tensor()
        self.pred_label = torch.Tensor()
        self.num_observed = torch.zeros(1)
        self.value = (self.label == self.pred_label).sum()/self.num_observed

class BinaryAccuracy(Metric):
    def __init__(self):
        self.reset()
    
    def update(self, label:torch.Tensor, prediction:torch.Tensor):
        # `prediction` must be probability
        pred_label = (prediction >= 0.5).to(dtype=torch.float)
        self.num_observed += label.shape[0]

        self.label = torch.cat(tensors=[self.label, label], dim=0)
        self.pred_label = torch.cat(tensors=[self.pred_label, pred_label], dim=0)

        self.value = (self.label == self.pred_label).sum()/self.num_observed
        return self.value

    def reset(self):
        self.label = torch.Tensor()
        self.pred_label = torch.Tensor()
        self.num_observed = torch.zeros(1)
        self.value = (self.label == self.pred_label).sum()/self.num_observed

if __name__ == '__main__':
    mean = Mean()
    accuracy = CategoricalAccuracy()
    print(mean, accuracy)