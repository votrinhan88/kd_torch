# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from typing import List

import torch
import numpy as np

from models.GANs.GAN import GAN
from utils.modules import Reshape

class DC_Generator(torch.nn.Module):
    """Generator for DCGAN.
    
    Args:
        `latent_dim`: Dimension of latent space. Defaults to `100`.
        `image_dim`: Dimension of synthetic images. Defaults to `[1, 28, 28]`.
        `base_dim`: Dimension of the shallowest feature maps. After each
            convolutional layer, each dimension is doubled the and number of filters
            is halved until `image_dim` is reached. Defaults to `[256, 7, 7]`.
    
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#generative-adversarial-networks
    """
    def __init__(self,
                 latent_dim:int=100,
                 image_dim:List[int]=[1, 28, 28],
                 base_dim:List[int]=[256, 7, 7],
                 ):
        """Initialize generator.
        
        Args:
            `latent_dim`: Dimension of latent space. Defaults to `100`.
            `image_dim`: Dimension of synthetic images. Defaults to `[1, 28, 28]`.
            `base_dim`: Dimension of the shallowest feature maps. After each
                convolutional layer, each dimension is doubled the and number of filters
                is halved until `image_dim` is reached. Defaults to `[256, 7, 7]`.
        """
        # Parse architecture from input dimension
        dim_ratio = [image_dim[axis]/base_dim[axis] for axis in torch.arange(start=1, end=len(image_dim))]
        for axis in range(len(dim_ratio)):
            num_conv = np.log(dim_ratio[axis])/np.log(2.)
            assert num_conv == int(num_conv), f'`base_dim` {base_dim[axis]} is not applicable with `image_dim` {image_dim[axis]} at axis {axis}.'
            assert dim_ratio[axis] == dim_ratio[0], f'Ratio of `image_dim` and `base_dim` {image_dim[axis]}/{base_dim[axis]} at axis {axis} is mismatched with {image_dim[0]}/{base_dim[0]} at axis 0.'
        num_conv = int(num_conv)

        super().__init__()
        self.latent_dim = latent_dim
        self.image_dim = image_dim
        self.base_dim = base_dim

        self.dense_0 = torch.nn.Linear(in_features=self.latent_dim, out_features=np.prod(self.base_dim), bias=False)
        self.reshape = Reshape(out_shape=self.base_dim)
        self.bnorm_0 = torch.nn.BatchNorm2d(num_features=self.base_dim[0])
        self.relu_0  = torch.nn.ReLU()

        convt_blocks = [None for i in range(num_conv)]
        for i in range(num_conv):
            in_channels = self.base_dim[0] // 2**i
            out_channels = self.base_dim[0] // 2**(i+1)
            if i < num_conv - 1:
                block = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    torch.nn.BatchNorm2d(num_features=out_channels),
                    torch.nn.ReLU(),
                )
            elif i == num_conv - 1:
                # Last Conv2DTranspose: not use BatchNorm, replace relu with tanh
                block = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=self.image_dim[0], kernel_size=4, stride=2, padding=1, bias=False),
                    torch.nn.Tanh()
                )
            convt_blocks[i] = block
        self.convt_blocks = torch.nn.Sequential(*convt_blocks)

    def forward(self, x):
        x = self.dense_0(x)
        x = self.reshape(x)
        x = self.bnorm_0(x)
        x = self.relu_0(x)
        x = self.convt_blocks(x)
        return x

class DC_Discriminator(torch.nn.Module):
    """Discriminator for DCGAN. Ideally should have a symmetric architecture with the
    generator's.

    Args:
        `image_dim`: Dimension of image. Defaults to `[1, 28, 28]`.
        `base_dim`: Dimension of the shallowest feature maps, ideally equal to the
            generator's. Opposite to the generator, after each convolutional layer,
            each dimension from `image_dim` is halved and the number of filters is
            doubled until `base_dim` is reached. Defaults to `[256, 7, 7]`.
        `return_logits`: flag to choose between return logits or probability.
            Defaults to `True`.
    
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#generative-adversarial-networks
    """    
    def __init__(self,
                 image_dim:List[int]=[1, 28, 28],
                 base_dim:List[int]=[256, 7, 7],
                 return_logits:bool=True,
                 ):
        """Initialize discriminator.
        
        Args:
            `image_dim`: Dimension of image. Defaults to `[1, 28, 28]`.
            `base_dim`: Dimension of the shallowest feature maps, ideally equal to the
                generator's. Opposite to the generator, after each convolutional layer,
                each dimension from `image_dim` is halved and the number of filters is
                doubled until `base_dim` is reached. Defaults to `[256, 7, 7]`.
            `return_logits`: flag to choose between return logits or probability.
                Defaults to `True`.
        """
        # Parse architecture from input dimension
        dim_ratio = [image_dim[axis]/base_dim[axis] for axis in torch.arange(start=1, end=len(image_dim))]
        for axis in range(len(dim_ratio)):
            num_conv = np.log(dim_ratio[axis])/np.log(2.)
            assert num_conv == int(num_conv), f'`base_dim` {base_dim[axis]} is not applicable with `image_dim` {image_dim[axis]} at axis {axis}.'
            assert dim_ratio[axis] == dim_ratio[0], f'Ratio of `image_dim` and `base_dim` {image_dim[axis]}/{base_dim[axis]} at axis {axis} is mismatched with {image_dim[0]}/{base_dim[0]} at axis 0.'
        num_conv = int(num_conv)
        
        super().__init__()
        self.image_dim = image_dim
        self.base_dim = base_dim
        self.return_logits = return_logits

        self.conv_blocks = [None for i in range(num_conv)]
        for i in range(num_conv):
            in_channels = self.base_dim[0] // 2**(num_conv-i)
            out_channels = self.base_dim[0] // 2**(num_conv-1-i)
            if i == 0:
                # First Conv2D: not use BatchNorm 
                self.conv_blocks[i] = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=self.image_dim[0], out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    torch.nn.LeakyReLU(negative_slope=0.2),
                )
            elif i > 0:
                self.conv_blocks[i] = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    torch.nn.BatchNorm2d(num_features=out_channels),
                    torch.nn.LeakyReLU(negative_slope=0.2)
                )
        self.conv_blocks = torch.nn.Sequential(*self.conv_blocks)

        self.flatten = torch.nn.Flatten()
        self.logits = torch.nn.Linear(in_features=np.prod(self.base_dim), out_features=1, bias=False)
        if self.return_logits is False:
            self.pred = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.flatten(x)
        x = self.logits(x)
        if self.return_logits is False:
            x = self.pred(x)
        return x

class DCGAN(GAN):
    """Unsupervised Representation Learning with Deep Convolutional Generative
    Adversarial Networks
    DOI: 10.48550/arXiv.1511.06434
    """    
    pass

if __name__ == '__main__':
    from torchinfo import summary
    from utils.dataloader import get_dataloader
    from utils.callbacks import CSVLogger
    from models.GANs.utils import MakeSyntheticGIFCallback

    def test_mnist():
        LATENT_DIM = 100
        IMAGE_DIM = [1, 28, 28]
        BASE_DIM = [256, 7, 7]
        BATCH_SIZE = 128
        NUM_EPOCHS = 50

        dataloader = get_dataloader(
            dataset='MNIST',
            resize=IMAGE_DIM[1:],
            rescale=[-1, 1],
            batch_size_train=BATCH_SIZE
        )

        gen = DC_Generator(
            latent_dim=LATENT_DIM,
            image_dim=IMAGE_DIM,
            base_dim=BASE_DIM)
        crit = DC_Discriminator(
            image_dim=IMAGE_DIM,
            base_dim=BASE_DIM,
            return_logits=False
        )

        summary(model=gen, input_size=[BATCH_SIZE, LATENT_DIM])
        summary(model=crit, input_size=[BATCH_SIZE, *IMAGE_DIM])

        gan = DCGAN(generator=gen, critic=crit)
        gan.compile(
            optimizer_gen=torch.optim.Adam(params=gen.parameters(), lr=2e-4, betas=(0.5, 0.999)),
            optimizer_crit=torch.optim.Adam(params=crit.parameters(), lr=2e-4, betas=(0.5, 0.999)),
            loss_fn=torch.nn.BCELoss()
        )

        csv_logger = CSVLogger(
            filename=f'./logs/{gan.__class__.__name__}.csv',
            append=True
        )
        gif_maker = MakeSyntheticGIFCallback(
            filename=f'./logs/{gan.__class__.__name__}.gif',
            nrows=5, ncols=5,
            postprocess_fn=lambda x:(x+1)/2
        )
        gan.training_loop(
            trainloader=dataloader['train'],
            num_epochs=NUM_EPOCHS,
            valloader=dataloader['val'],
            callbacks=[csv_logger, gif_maker],
        )
    
    test_mnist()