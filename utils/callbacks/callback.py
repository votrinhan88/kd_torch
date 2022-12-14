from abc import ABC

class Callback(ABC):
    """Base class for callbacks."""
    def hook(self, host):
        self.host = host
        self.device = self.host.device
        
    def on_train_begin(self, logs:dict=None): return
    def on_train_end(self, logs:dict=None): return
    def on_test_begin(self, logs:dict=None): return
    def on_test_end(self, logs:dict=None): return

    def on_epoch_begin(self, epoch:int, logs:dict=None): return
    def on_epoch_train_end(self, epoch:int, logs:dict=None): return
    def on_epoch_test_begin(self, epoch:int, logs:dict=None): return
    def on_epoch_end(self, epoch:int, logs:dict=None): return
    def on_train_batch_begin(self, batch:int, logs:dict=None): return
    def on_train_batch_end(self, batch:int, logs:dict=None): return
    def on_test_batch_begin(self, batch:int, logs:dict=None): return
    def on_test_batch_end(self, batch:int, logs:dict=None): return