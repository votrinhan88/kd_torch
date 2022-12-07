import copy
from typing import Callable, Tuple, Optional, Dict
import torch
from torch.utils.data import DataLoader
import tqdm.auto as tqdm

if __name__ == '__main__':
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
    sys.path.append(repo_path)
from metrics.Metrics import Metric, Mean, CategoricalAccuracy

class Trainer:
    def __init__(self,
                 model:torch.nn.Module,
                 optimizer:torch.optim.Optimizer,
                 loss_fn:Callable
                 ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        self.train_metrics:Dict[str, Metric] = {'loss': Mean(), 'accuracy': CategoricalAccuracy()}
        self.val_metrics = copy.deepcopy(self.train_metrics)

        self.history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
        }

    def fit(self,
                      trainloader:DataLoader,
                      num_epochs:int,
                      valloader:Optional[DataLoader],
                      ):
        # on_train_begin
        training_loop = tqdm.tqdm(range(num_epochs), desc='Fit', unit='epochs')
        for epoch in training_loop:
            # on_epoch_begin
            
            # Training phase
            train_data = tqdm.tqdm(trainloader, desc='Train phase', leave=False, unit='batches')
            for batch, data in enumerate(train_data):
                # on_train_batch_begin
                self.train_batch(data)
                # on_train_batch_end
                train_data.set_postfix({
                    'loss':f"{self.train_metrics['loss'].latest:.4g}",
                    'accuracy':f"{self.train_metrics['accuracy'].latest:.4g}",
                })
            postfix_training_loop = {
                'loss':f"{self.train_metrics['loss'].latest:.4g}",
                'accuracy':f"{self.train_metrics['accuracy'].latest:.4g}",
            }
            training_loop.set_postfix(postfix_training_loop)
            
            # Validation phase
            if valloader is not None:
                # on_test_begin
                val_data = tqdm.tqdm(valloader, desc='Test phase', leave=False, unit='batches')
                for batch, data in enumerate(val_data):
                    # on_test_batch_begin
                    self.test_batch(data)
                    # on_test_batch_end
                    val_data.set_postfix({
                        'loss':f"{self.val_metrics['loss'].latest:.4g}",
                        'accuracy':f"{self.val_metrics['accuracy'].latest:.4g}",
                    })
                # on_test_end
                postfix_training_loop.update({
                    'val_loss':f"{self.val_metrics['loss'].latest:.4g}",
                    'val_accuracy':f"{self.val_metrics['accuracy'].latest:.4g}",
                })
                training_loop.set_postfix(postfix_training_loop)

            # on_epoch_end
            for metric in [*self.train_metrics.values(), *self.val_metrics.values()]:
                metric.reset()
        # on_train_end
        return self.history

    def train_batch(self, data:Tuple[torch.Tensor, torch.Tensor]):
        input, label = data
        # Forward
        prediction = self.model(input)
        loss = self.loss_fn(prediction, label)
        # Backward
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Metrics
        with torch.inference_mode():
            self.train_metrics['loss'].update(new_entry=loss)
            self.train_metrics['accuracy'].update(label=label, prediction=prediction)

    def test_batch(self, data:Tuple[torch.Tensor, torch.Tensor]):
        input, label = data
        with torch.inference_mode():
            # Forward
            prediction = self.model(input)
            loss = self.loss_fn(prediction, label)
            self.val_metrics['loss'].update(new_entry=loss)
            self.val_metrics['accuracy'].update(label=label, prediction=prediction)