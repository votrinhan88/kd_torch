from typing import List
import torch

class LeNet5(torch.nn.Module):
    '''Gradient-based learning applied to document recognition
    DOI: 10.1109/5.726791

    Args:
        `half`: Flag to choose between LeNet-5 or LeNet-5-Half. Defaults to `False`.
        `input_dim`: Dimension of input images. Defaults to `[1, 32, 32]`.
        `num_classes`: Number of output nodes. Defaults to `10`.
        `return_logits`: Flag to choose between return logits or probability.
            Defaults to `True`.

    Two versions: LeNet-5 and LeNet-5-Half
    '''
    def __init__(self,
                 half_size:bool=False,
                 input_dim:List[int]=[1, 32, 32],
                 num_classes:int=10,
                 ActivationLayer=torch.nn.Tanh,
                 PoolLayer=torch.nn.AvgPool2d,
                 return_logits:bool=True,
                 ):
        """Initialize model.

        Args:
            `half`: Flag to choose between LeNet-5 or LeNet-5-Half. Defaults to `False`.
            `input_dim`: Dimension of input images. Defaults to `[1, 32, 32]`.
            `num_classes`: Number of output nodes. Defaults to `10`.
            `return_logits`: Flag to choose between return logits or probability.
                Defaults to `True`.
        """        
        assert isinstance(half_size, bool), '`half_size` must be of type bool'
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'
        
        super().__init__()
        self.half_size = half_size
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.ActivationLayer = ActivationLayer
        self.PoolLayer = PoolLayer
        self.return_logits = return_logits

        if self.half_size is False:
            divisor = 1
        elif self.half_size is True:
            divisor = 2
        
        # Layers: C: convolutional, A: activation, S: pooling
        self.C1 = torch.nn.Conv2d(in_channels=self.input_dim[0], out_channels=6//divisor, kernel_size=5, stride=1, padding=0)
        self.A1 = self.ActivationLayer()
        self.S2 = self.PoolLayer(kernel_size=2, stride=2, padding=0)
        self.C3 = torch.nn.Conv2d(in_channels=6//divisor, out_channels=16//divisor, kernel_size=5, stride=1, padding=0)
        self.A3 = self.ActivationLayer()
        self.S4 = self.PoolLayer(kernel_size=2, stride=2, padding=0)
        self.C5 = torch.nn.Conv2d(in_channels=16//divisor, out_channels=120//divisor, kernel_size=5, stride=1, padding=0)
        self.A5 = self.ActivationLayer()
        self.flatten = torch.nn.Flatten()
        self.F6 = torch.nn.Linear(in_features=120//divisor, out_features=84//divisor)
        self.A6 = self.ActivationLayer()
        self.logits = torch.nn.Linear(in_features=84//divisor, out_features=self.num_classes)
        if self.return_logits is False:
            if self.num_classes == 1:
                self.pred = torch.nn.Sigmoid()
            elif self.num_classes > 1:
                self.pred = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.C1(x)
        x = self.A1(x)
        x = self.S2(x)
        x = self.C3(x)
        x = self.A3(x)
        x = self.S4(x)
        x = self.C5(x)
        x = self.A5(x)
        x = self.flatten(x)
        x = self.F6(x)
        x = self.A6(x)
        x = self.logits(x)
        if self.return_logits is False:
            x = self.pred(x)
        return x

if __name__ == '__main__':
    # Change path
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
    if sys.path[0] != repo_path:
        sys.path.insert(0, repo_path)
    
    from dataloader import get_dataloader
    from models.classifiers.utils import ClassifierTrainer
    from utils.callbacks import CSVLogger
    
    def test_mnist():
        IMAGE_DIM = [1, 32, 32]
        NUM_CLASSES = 10
        NUM_EPOCHS = 10

        dataloader = get_dataloader(
            dataset='MNIST',
            resize=IMAGE_DIM[1:],
            rescale=[-1, 1],
        )

        net = LeNet5(
            half_size=False,
            input_dim=IMAGE_DIM,
            num_classes=NUM_CLASSES,
            ActivationLayer=torch.nn.ReLU,
            PoolLayer=torch.nn.MaxPool2d,
        )

        trainer = ClassifierTrainer(model=net)
        trainer.compile(
            optimizer=torch.optim.Adam(params=net.parameters(), lr=1e-3),
            loss_fn=torch.nn.CrossEntropyLoss())

        csv_logger = CSVLogger(filename=f'./logs/{net.__class__.__name__}.csv', append=True)
        trainer.training_loop(
            trainloader=dataloader['train'],
            num_epochs=NUM_EPOCHS,
            valloader=dataloader['val'],
            callbacks=[csv_logger],
        )

    test_mnist()