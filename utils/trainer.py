# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from abc import ABC, abstractmethod
from typing import Sequence, Optional, Dict, List

import torch
from torch.utils.data import DataLoader
import tqdm.auto as tqdm

from utils.metrics import Metric
from utils.callbacks import Callback, History, ProgressBar

class Trainer(ABC):
    @abstractmethod
    def __init__(self,
                 device:Optional[str]=None):
        # Parse device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    @abstractmethod
    def compile(self):
        # Metrics
        self.train_metrics:Dict[str, Metric] = {}
        self.val_metrics:Dict[str, Metric] = {}
        # To be initialize by callback History
        self.history:History
        # To be initialize by callback ProgressBar
        self.training_progress:tqdm = None
        self.train_phase_progress:tqdm = None
        self.val_phase_progress:tqdm = None

    def training_loop(self,
                      trainloader:DataLoader,
                      num_epochs:int,
                      valloader:Optional[DataLoader]=None,
                      callbacks:Optional[List[Callback]]=None,
                      ):
        self.training_loop_kwargs = {
            'trainloader':trainloader,
            'num_epochs':num_epochs,
            'valloader':valloader,
        }

        self.hook_callbacks(callbacks=callbacks)
        logs = {}
        self.on_train_begin(logs)
        
        for epoch in self.training_progress:
            epoch_logs = {}
            self.on_epoch_begin(epoch_logs, logs)
            
            # Training phase
            for batch, data in enumerate(self.train_phase_progress):
                batch_logs = {}
                self.on_train_batch_begin(batch, batch_logs)
                self.train_batch(data)
                batch_logs.update({k:v.latest for (k, v) in self.train_metrics.items()})
                self.on_train_batch_end(batch, batch_logs)
            epoch_logs.update(batch_logs)
            self.on_epoch_train_end(epoch, epoch_logs)
            
            # Validation phase
            if valloader is not None:
                self.on_epoch_test_begin(epoch, epoch_logs)
                for batch, data in enumerate(self.val_phase_progress):
                    batch_logs = {}
                    self.on_test_batch_begin(batch, batch_logs)
                    self.test_batch(data)
                    batch_logs.update({k:v.latest for (k, v) in self.val_metrics.items()})
                    self.on_test_batch_end(batch, batch_logs)
                epoch_logs.update({f'val_{key}': value for key, value in batch_logs.items()})
            self.on_epoch_end(epoch, epoch_logs)
        
        logs.update(epoch_logs)
        self.on_train_end(logs)

        return self.history

    @abstractmethod
    def train_batch(self, data:Sequence[torch.Tensor]): pass

    @abstractmethod
    def test_batch(self, data:Sequence[torch.Tensor]): pass

    def hook_callbacks(self, callbacks:List[Callback]):
        self.callbacks:List[Callback] = [History(), ProgressBar()]
        if callbacks is not None:
            self.callbacks.extend(callbacks)
        for cb in self.callbacks:
            cb.hook(self)

    def on_train_begin(self, logs=None):
        for cb in self.callbacks:
            cb.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for cb in self.callbacks:
            cb.on_train_end(logs)

    def on_epoch_begin(self, epoch:int, logs=None):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)
        # Reset metrics
        for metric in [*self.train_metrics.values(), *self.val_metrics.values()]:
            metric.reset()
    
    def on_epoch_train_end(self, epoch:int, logs=None):
        for cb in self.callbacks:
            cb.on_epoch_train_end(epoch, logs)
    
    def on_epoch_test_begin(self, epoch:int, logs=None):
        for cb in self.callbacks:
            cb.on_epoch_test_begin(epoch, logs)

    def on_epoch_end(self, epoch:int, logs=None):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)

    def on_train_batch_begin(self, batch:int, logs=None):
        for cb in self.callbacks:
            cb.on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch:int, logs=None):
        for cb in self.callbacks:
            cb.on_train_batch_end(batch, logs)

    def on_test_batch_begin(self, batch:int, logs=None):
        for cb in self.callbacks:
            cb.on_test_batch_begin(batch, logs)

    def on_test_batch_end(self, batch:int, logs=None):
        for cb in self.callbacks:
            cb.on_test_batch_end(batch, logs)