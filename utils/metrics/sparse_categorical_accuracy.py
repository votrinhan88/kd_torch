import torch
from .metric import Metric

class SparseCategoricalAccuracy(Metric):
    def __init__(self):
        self.reset()
    
    def update(self, prediction:torch.Tensor, label:torch.Tensor):
        label = label.to(device='cpu')
        prediction = prediction.to(device='cpu')

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