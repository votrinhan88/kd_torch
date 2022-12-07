from abc import ABC
import collections
import csv
import os
import numpy as np

class Callback(ABC):
    """Base class for callbacks."""
    def hook(self, trainer):
        self.trainer = trainer
        
    def on_train_begin(self, logs=None): pass
    def on_train_end(self, logs=None): pass
    def on_epoch_begin(self, epoch, logs=None): pass
    def on_epoch_end(self, epoch, logs=None): pass

class HistoryCallback(Callback):
    def __init__(self):
        super().__init__()
        self.history = {}
        self.metrics = {}

    def on_train_begin(self, logs=None):
        for (key, metric) in self.trainer.train_metrics.items():
            self.metrics.update({key: metric})
            self.history.update({key: []})
        for (key, metric) in self.trainer.val_metrics.items():
            self.metrics.update({f'val_{key}': metric})
            self.history.update({f'val_{key}': []})

    def on_epoch_end(self, epoch:int, logs=None):
        for (key, metric) in self.metrics.items():
            self.history[key].append(metric.latest)
    
    def on_train_end(self, logs=None):
        self.trainer.history = self.history

class CSVLogger(Callback):
    def __init__(self, filename:str, separator:str=",", append:bool=False):
        super().__init__()
        self.filename = filename
        self.separator = separator
        self.append = append
        # self.writer = None
        # self.keys = None
        # self.append_header = True

    def check_header(filename):
        with open(filename) as f:
            first = f.read(1)
        return first not in '.-0123456789'

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.isfile(self.filename):
                with open(self.filename, "r") as f:
                    self.append_header = not bool(len(f.readline()))
            mode = "a"
        else:
            mode = "w"
        self.csv_file = open(self.filename, mode)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

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

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict(
                (k, logs[k]) if k in logs else (k, "NA") for k in self.keys
            )

        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.separator

            fieldnames = ["epoch"] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
            )
            if self.append_header:
                self.writer.writeheader()

        row_dict = collections.OrderedDict({"epoch": epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None
    