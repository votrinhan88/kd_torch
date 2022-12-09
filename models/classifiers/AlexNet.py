from typing import List

import torch
import numpy as np

class AlexNet(torch.nn.Module):
    """ImageNet classification with deep convolutional neural networks - Krizhevsky
    et al. (2012)
    DOI: 10.1145/3065386

    Args:
        `half_size`: Flag to choose between AlexNet or AlexNet-Half. Defaults to `False`.
        `input_dim`: Dimension of input images. Defaults to `[3, 32, 32]`.
        `num_classes`: Number of output nodes. Defaults to `10`.
        `return_logits`: Flag to choose between return logits or probability.
            Defaults to `True`.
    
    Two versions: AlexNet and AlexNet-Half following the architecture in 'Zero-Shot
    Knowledge Distillation in Deep Networks' - Nayak et al. (2019)
    Implementation: https://github.com/nphdang/FS-BBT/blob/main/cifar10/alexnet_model.py
    """    
    def __init__(self,
                 half_size:bool=False,
                 input_dim:List[int]=[3, 32, 32],
                 num_classes:int=10,
                 return_logits:bool=True,
                 ):
        """Initialize model.
        
        Args:
            `half_size`: Flag to choose between AlexNet or AlexNet-Half. Defaults to `False`.
            `input_dim`: Dimension of input images. Defaults to `[3, 32, 32]`.
            `num_classes`: Number of output nodes. Defaults to `10`.
            `return_logits`: Flag to choose between return logits or probability.
                Defaults to `True`.
        
        Kwargs:
            Additional keyword arguments passed to `torch.nn.Model.__init__`.
        """    
        assert isinstance(half_size, bool), "'half' should be of type bool"
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'
        super().__init__()

        if half_size is False:
            divisor = 1
        elif half_size is True:
            divisor = 2

        self.half_size = half_size
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.return_logits = return_logits
        
        # Convolutional blocks
        ongoing_shape = self.input_dim
        self.conv_1 = torch.nn.Sequential(
            # bias_initializer='zeros'
            torch.nn.Conv2d(in_channels=self.input_dim[0], out_channels=48//divisor, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            torch.nn.BatchNorm2d(num_features=48//divisor)
        )
        ongoing_shape = [48//divisor, *[(dim - 1)//2 for dim in ongoing_shape[1:]]]
        self.conv_2 = torch.nn.Sequential(
            # bias_initializer='zeros'
            torch.nn.Conv2d(in_channels=48//divisor, out_channels=128//divisor, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            torch.nn.BatchNorm2d(num_features=128//divisor)
        )
        # [None, 128, 6, 6]
        ongoing_shape = [128//divisor, *[(dim - 1)//2 for dim in ongoing_shape[1:]]]
        self.conv_3 = torch.nn.Sequential(
            # bias_initializer='zeros'
            torch.nn.Conv2d(in_channels=128//divisor, out_channels=192//divisor, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=192//divisor)
        )
        ongoing_shape = [192//divisor, *ongoing_shape[1:]]
        self.conv_4 = torch.nn.Sequential(
            # bias_initializer='zeros'
            torch.nn.Conv2d(in_channels=192//divisor, out_channels=192//divisor, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=192//divisor)
        )
        ongoing_shape = [192//divisor, *ongoing_shape[1:]]
        self.conv_5 = torch.nn.Sequential(
            # bias_initializer='zeros'
            torch.nn.Conv2d(in_channels=192//divisor, out_channels=128//divisor, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            torch.nn.BatchNorm2d(num_features=128//divisor)
        )
        ongoing_shape = [128//divisor, *[(dim - 1)//2 for dim in ongoing_shape[1:]]]
        self.flatten = torch.nn.Flatten()
        ongoing_shape = [np.prod(ongoing_shape)]
        # Fully-connected layers
        self.fc_1 = torch.nn.Sequential(
            # bias_initializer='zeros'
            torch.nn.Linear(in_features=ongoing_shape[0], out_features=512//divisor),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.BatchNorm1d(num_features=512//divisor)
        )
        ongoing_shape = [512//divisor]
        self.fc_2 = torch.nn.Sequential(
            # bias_initializer='zeros'
            torch.nn.Linear(in_features=ongoing_shape[0], out_features=256//divisor),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.BatchNorm1d(num_features=256//divisor)
        )
        ongoing_shape = [256//divisor]
        self.logits = torch.nn.Linear(in_features=ongoing_shape[0], out_features=self.num_classes)
        if self.return_logits is False:
            if self.num_classes == 1:
                self.pred = torch.nn.Sigmoid()
            elif self.num_classes > 1:
                self.pred = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.logits(x)
        if self.return_logits is False:
            x = self.pred(x)
        return x

if __name__ == '__main__':
    # Change path
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
    sys.path.append(repo_path)

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
            batch_size_train=NUM_EPOCHS,
        )

        net = AlexNet(input_dim=IMAGE_DIM, num_classes=NUM_CLASSES)
        summary(model=net, input_size=[BATCH_SIZE, *IMAGE_DIM])
        
        trainer = Trainer(
            model=net,
            optimizer=torch.optim.Adam(params=net.parameters(), lr=1e-3),
            loss_fn=torch.nn.CrossEntropyLoss()
        )
        csv_logger = CSVLogger(filename=f'./logs/{net.__class__.__name__}.csv', append=True)
        trainer.training_loop(
            trainloader=dataloader['train'],
            num_epochs=NUM_EPOCHS,
            valloader=dataloader['val'],
            callbacks=[csv_logger],
        )

    test_mnist()