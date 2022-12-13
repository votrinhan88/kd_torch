# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from typing import Callable, Tuple, Optional, Any
import torch
from utils.trainer import Trainer
from utils.metrics import Mean, CategoricalAccuracy

class ClassifierTrainer(Trainer):
    def __init__(self,
                 model:torch.nn.Module,
                 device:Optional[str]=None):
        super().__init__(device=device)
        self.model = model.to(self.device)

    def compile(self,
                optimizer:torch.optim.Optimizer,
                loss_fn:Callable[[Any], torch.Tensor]):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        # Metrics
        self.train_metrics = {'loss': Mean(), 'accuracy': CategoricalAccuracy()}
        self.val_metrics = {'loss': Mean(), 'accuracy': CategoricalAccuracy()}

    def train_batch(self, data:Tuple[torch.Tensor, torch.Tensor]):
        # Unpack data
        input, label = data
        input, label = input.to(self.device), label.to(self.device)
        
        self.model.train()
        self.optimizer.zero_grad()
        # Forward
        prediction = self.model(input)
        loss = self.loss_fn(prediction, label)
        # Backward
        loss.backward()
        self.optimizer.step()
        with torch.inference_mode():
            # Metrics
            self.train_metrics['loss'].update(new_entry=loss)
            self.train_metrics['accuracy'].update(label=label, prediction=prediction)

    def test_batch(self, data:Tuple[torch.Tensor, torch.Tensor]):
        # Unpack data
        input, label = data
        input, label = input.to(self.device), label.to(self.device)
        
        self.model.eval()
        with torch.inference_mode():
            # Forward
            prediction = self.model(input)
            loss = self.loss_fn(prediction, label)
            # Metrics
            self.val_metrics['loss'].update(new_entry=loss)
            self.val_metrics['accuracy'].update(label=label, prediction=prediction)