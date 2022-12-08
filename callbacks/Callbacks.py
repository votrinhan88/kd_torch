from abc import ABC
import collections
import csv
import os
import numpy as np
import tqdm.auto as tqdm

class Callback(ABC):
    """Base class for callbacks."""
    def hook(self, host):
        self.host = host
        
    def on_train_begin(self, logs=None): return
    def on_train_end(self, logs=None): return
    def on_epoch_begin(self, epoch:int, logs=None): return
    def on_epoch_train_end(self, epoch:int, logs=None): return
    def on_epoch_test_begin(self, epoch:int, logs=None): return
    def on_epoch_end(self, epoch:int, logs=None): return
    def on_train_batch_begin(self, batch:int, logs=None): return
    def on_train_batch_end(self, batch:int, logs=None): return
    def on_test_batch_begin(self, batch:int, logs=None): return
    def on_test_batch_end(self, batch:int, logs=None): return

class History(Callback):
    def __init__(self):
        super().__init__()
        self.history = {}
        self.metrics = {}

    def on_train_begin(self, logs=None):
        for (key, metric) in self.host.train_metrics.items():
            self.metrics.update({key: metric})
            self.history.update({key: []})
        for (key, metric) in self.host.val_metrics.items():
            self.metrics.update({f'val_{key}': metric})
            self.history.update({f'val_{key}': []})

    def on_epoch_end(self, epoch:int, logs=None):
        for (key, metric) in self.metrics.items():
            self.history[key].append(metric.latest)
    
    def on_train_end(self, logs=None):
        self.host.history = self

class CSVLogger(Callback):
    def __init__(self, filename:str, separator:str=",", append:bool=False):
        super().__init__()
        self.filename = filename
        self.separator = separator
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True

    @staticmethod
    def handle_value(k):
        is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
        if isinstance(k, str):
            return k
        elif (
            isinstance(k, collections.abc.Iterable)
            and not is_zero_dim_ndarray
        ):
            return f"\"[{', '.join(map(str, k))}]\""
        else:
            return k

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.isfile(self.filename):
                with open(file=self.filename, mode="r") as f:
                    self.append_header = not bool(len(f.readline()))
            mode = "a"
        else:
            mode = "w"
        self.csv_file = open(file=self.filename, mode=mode, newline='')

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        if self.keys is None:
            train_keys, val_keys = [], []
            for key in logs.keys():
                if key[0:4] == 'val_':
                    val_keys.append(key)
                else:
                    train_keys.append(key)
            self.keys = sorted(train_keys) + sorted(val_keys)

        # if self.model.stop_training:
        #     # We set NA so that csv parsers do not fail for this last epoch.
        #     logs = dict(
        #         (k, logs[k]) if k in logs else (k, "NA") for k in self.keys
        #     )
        

        if self.writer is None:
            class CustomDialect(csv.excel):
                delimiter = self.separator
            
            fieldnames = ["epoch"] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
            )
            if self.append_header:
                self.writer.writeheader()

        row_dict = collections.OrderedDict({"epoch": epoch})
        row_dict.update((key, self.handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None

class ProgressBar(Callback):
    def on_train_begin(self, logs=None):
        self.kwargs = self.host.training_loop_kwargs
        self.host.training_progress = tqdm.tqdm(range(self.kwargs['num_epochs']), desc='Training', unit='epochs')

    def on_epoch_train_end(self, epoch:int, logs=None):
        self.host.training_progress.set_postfix({k: f'{v:.4g}' for (k, v) in logs.items()})

    def on_epoch_end(self, epoch:int, logs=None):
        self.host.training_progress.set_postfix({k: f'{v:.4g}' for (k, v) in logs.items()})

    def on_epoch_begin(self, epoch:int, logs=None):
        self.host.train_phase_progress = tqdm.tqdm(self.kwargs['trainloader'], desc='Train phase', leave=False, unit='batches')

    def on_train_batch_end(self, batch:int, logs=None):
        self.host.train_phase_progress.set_postfix({k: f'{v:.4g}' for (k, v) in logs.items()})
        
    def on_epoch_test_begin(self, epoch:int, logs=None):
        self.host.val_phase_progress = tqdm.tqdm(self.kwargs['valloader'], desc='Test phase', leave=False, unit='batches')

    def on_test_batch_end(self, batch:int, logs=None):
        self.host.val_phase_progress.set_postfix({k: f'{v:.4g}' for (k, v) in logs.items()})