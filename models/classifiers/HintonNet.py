if __name__ == '__main__':
    # Change path
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
    sys.path.append(repo_path)

from typing import List, Union
import torch
import numpy as np
from callbacks.Callbacks import Callback

class HintonNet(torch.nn.Module):
    """Baseline model in implemented in paper 'Distilling the Knowledge in a Neural
    Network' - Hinton et al. (2015), DOI: 10.48550/arXiv.1503.02531. Originally
    described in Improving neural networks by preventing co-adaptation of
    feature detectors - Hinton et al (2012), DOI: 10.48550/arXiv.1207.0580

    Args:
        `input_dim`: Dimension of input images. Defaults to `[1, 28, 28]`.
        `hidden_layers`: Number of nodes in each hidden layer.
            Defaults to `[1200, 1200]`.
        `num_classes`: Number of output nodes. Defaults to `10`.
        `return_logits`: Flag to choose between return logits or probability.
            Defaults to `True`.

    Two versions:
    - Teacher: `hidden_layers` = [1200, 1200]
    - Student: `hidden_layers` = [800, 800]
    """    
    _name = 'HintonNet'

    def __init__(self,
                 input_dim:List[int]=[1, 28, 28],
                 hidden_layers:List[int]=[1200, 1200],
                 num_classes:int=10,
                 return_logits:bool=True,
                 ):
        """Initialize model.
        
        Args:
            `input_dim`: Dimension of input images. Defaults to `[1, 28, 28]`.
            `hidden_layers`: Number of nodes in each hidden layer.
                Defaults to `[1200, 1200]`.
            `num_classes`: Number of output nodes. Defaults to `10`.
            `return_logits`: Flag to choose between return logits or probability.
                Defaults to `True`.
        """
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'

        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.return_logits = return_logits

        self.flatten    = torch.nn.Flatten()
        self.dropout_in = torch.nn.Dropout(p=0.2)

        self.hidden = []
        for num_in, num_out in zip(
                [np.prod(input_dim), *self.hidden_layers[0:-1]],
                self.hidden_layers):
            self.hidden.extend([
                torch.nn.Linear(in_features=num_in, out_features=num_out),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5),
            ])
        self.hidden = torch.nn.Sequential(*self.hidden)

        self.logits = torch.nn.Linear(in_features=self.hidden_layers[-1], out_features=self.num_classes)
        if self.return_logits is False:
            if self.num_classes == 1:
                self.pred = torch.nn.Sigmoid()  
            elif self.num_classes > 1:
                self.pred = torch.nn.Softmax(dim=1)

        # Initialize weights
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0, std=0.01)
            
    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout_in(x)
        x = self.hidden(x)
        x = self.logits(x)
        if self.return_logits is False:
            x = self.pred(x)
        return x

class MaxNormConstraint(Callback):
    def __init__(self,
                 max_norm:float=15,
                 ord:Union[int, float]=2
                ):
        super().__init__()
        self.max_norm = max_norm
        self.ord = ord

    def constraint_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            norm = torch.linalg.norm(module.weight, ord=self.ord, dim=1).unsqueeze(dim=1)
            desired = torch.clamp(norm, max=self.max_norm)
            module.weight *= desired / norm

    def on_train_batch_end(self, batch: int, logs=None):
        with torch.no_grad():
            self.host.model.apply(self.constraint_weights)

if __name__ == '__main__':
    from torchinfo import summary
    from dataloader import get_dataloader
    from models.classifiers.utils import Trainer
    from callbacks.Callbacks import CSVLogger

    def test_mnist():
        IMAGE_DIM = [1, 28, 28]
        NUM_CLASSES = 10
        NUM_EPOCHS = 10
        BATCH_SIZE = 64

        dataloader = get_dataloader(
            dataset='MNIST',
            rescale=[-1, 1],
            batch_size_train=BATCH_SIZE
        )

        net = HintonNet(
            input_dim=IMAGE_DIM,
            hidden_layers=[1200, 1200],
            num_classes=NUM_CLASSES,
        )
        summary(model=net, input_size=[BATCH_SIZE, *IMAGE_DIM])

        trainer = Trainer(
            model=net,
            optimizer=torch.optim.Adam(params=net.parameters(), lr=1e-3),
            loss_fn=torch.nn.CrossEntropyLoss()
        )
        csv_logger = CSVLogger(filename=f'./logs/{net.__class__.__name__}.csv', append=True)
        weight_constraint = MaxNormConstraint()
        trainer.training_loop(
            trainloader=dataloader['train'],
            num_epochs=NUM_EPOCHS,
            valloader=dataloader['val'],
            callbacks=[csv_logger, weight_constraint],
        )

    test_mnist()