from abc import ABC, abstractmethod
import torch

class Metric(ABC):
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
    @abstractmethod
    def latest(self):
        """Return the latest metric value in a pythonic format."""
        return self.value.item()

class Mean(Metric):
    def __init__(self):
        self.init_val = torch.zeros(1)
        self.reset()

    def update(self, new_entry:torch.Tensor) -> torch.Tensor:
        self.step += 1
        self.value = (
            self.value*self.step/(self.step + 1) +
                       new_entry/(self.step + 1)
        )
        return self.value

    def reset(self):
        self.step:int = -1
        self.value = self.init_val

    @property
    def latest(self):
        return super().latest

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

    @property
    def latest(self):
        return super().latest