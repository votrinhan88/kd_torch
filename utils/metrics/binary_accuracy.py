import torch
from .metric import Metric

class BinaryAccuracy(Metric):
    def __init__(self):
        self.reset()
    
    def update(self, label:torch.Tensor, prediction:torch.Tensor):
        label = label.to('cpu')
        prediction = prediction.to('cpu')

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