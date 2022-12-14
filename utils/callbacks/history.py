from .callback import Callback

class History(Callback):
    def on_train_begin(self, logs=None):
        self.reset()
        for (key, metric) in self.host.train_metrics.items():
            self.metrics.update({key: metric})
            self.history.update({key: []})
        for (key, metric) in self.host.val_metrics.items():
            self.metrics.update({f'val_{key}': metric})
            self.history.update({f'val_{key}': []})

    def on_train_end(self, logs=None):
        self.host.history = self

    def on_test_begin(self, logs=None):
        self.reset()
        for (key, metric) in self.host.val_metrics.items():
            self.metrics.update({f'val_{key}': metric})
            self.history.update({f'val_{key}': []})
    
    def on_test_end(self, logs=None):
        self.on_epoch_end(epoch=0, logs=logs)
        self.host.history = self

    def on_epoch_end(self, epoch:int, logs=None):
        for (key, metric) in self.metrics.items():
            self.history[key].append(metric.latest)

    def reset(self):
        self.history = {}
        self.metrics = {}