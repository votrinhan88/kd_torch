import tqdm.auto as tqdm

from .callback import Callback

class ProgressBar(Callback):
    def on_train_begin(self, logs=None):
        self.kwargs = self.host.training_loop_kwargs
        self.host.training_progress = tqdm.tqdm(range(self.kwargs['num_epochs']), desc='Training', unit='epochs')

    def on_test_begin(self, logs=None):
        self.kwargs = self.host.evaluate_kwargs
        self.host.val_phase_progress = tqdm.tqdm(self.kwargs['valloader'], desc='Evaluating', unit='batches')
    
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
