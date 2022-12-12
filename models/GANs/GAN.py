from typing import List, Tuple, Any, Callable, Optional, Dict
import torch
import numpy as np
from tqdm.auto import tqdm

if __name__ == '__main__':
    # Change path
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
    sys.path.append(repo_path)

from models.classifiers.utils import Trainer
from metrics.Metrics import Metric, Mean, BinaryAccuracy
from callbacks.Callbacks import History
from models.GANs.utils import Reshape, MakeSyntheticGIFCallback

class Generator(torch.nn.Module):
    """Generator for Generative Adversarial Networks.
            
    Args:
        `latent_dim`: Dimension of latent space. Defaults to `100`.
        `image_dim`: Dimension of synthetic images. Defaults to `[1, 28, 28]`.

    https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py
    """
    def __init__(self,
                 latent_dim:int=100,
                 image_dim:List[int]=[1, 28, 28],
                 ):
        """Initialize generator.
        
        Args:
            `latent_dim`: Dimension of latent space. Defaults to `100`.
            `image_dim`: Dimension of synthetic images. Defaults to `[1, 28, 28]`.
        """                
        super().__init__()
        self.latent_dim = latent_dim
        self.image_dim = image_dim

        def define_block(in_features:int, out_features:int):
            return torch.nn.Sequential(
                torch.nn.Linear(in_features=in_features, out_features=out_features),
                torch.nn.LeakyReLU(negative_slope=0.2),
                torch.nn.BatchNorm1d(num_features=out_features, momentum=0.8),
            )
        
        self.block_0  = define_block(in_features=self.latent_dim, out_features=256)
        self.block_1  = define_block(in_features=256, out_features=512)
        self.block_2  = define_block(in_features=512, out_features=1024)
        self.linear_3 = torch.nn.Linear(in_features=1024, out_features=np.prod(self.image_dim))
        self.reshape  = Reshape(out_shape=self.image_dim)
        self.tanh     = torch.nn.Tanh()
    
    def forward(self, x):
        x = self.block_0(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.linear_3(x)
        x = self.reshape(x)
        x = self.tanh(x)
        return x

class Discriminator(torch.nn.Module):
    """Discriminator for Generative Adversarial Networks.

    Args:
        `image_dim`: Dimension of input image. Defaults to `[1, 28, 28]`.
        `return_logits`: flag to choose between return logits or probability.
            Defaults to `True`.

    https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py
    """    
    def __init__(self,
                 image_dim:List[int]=[1, 28, 28],
                 return_logits:bool=True,
                ):
        """Initialize discriminator.
        
        Args:
            `image_dim`: Dimension of input image. Defaults to `[1, 28, 28]`.
            `return_logits`: flag to choose between return logits or probability.
                Defaults to `True`.
        """
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'
        super().__init__()
        self.image_dim = image_dim
        self.return_logits = return_logits

        self.flatten = torch.nn.Flatten()
        def define_block(in_features:int, out_features:int):
            return torch.nn.Sequential(
                torch.nn.Linear(in_features=in_features, out_features=out_features),
                torch.nn.LeakyReLU(negative_slope=0.2),
            )
        self.block_0 = define_block(in_features=np.prod(self.image_dim), out_features=512)
        self.block_1 = define_block(in_features=512, out_features=256)
        self.logits = torch.nn.Linear(in_features=256, out_features=1)
        if self.return_logits is False:
            self.pred = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.block_0(x)
        x = self.block_1(x)
        x = self.logits(x)
        if self.return_logits is False:
            x = self.pred(x)
        return x

class GAN(Trainer):
    """Generative Adversarial Networks.
    DOI: 10.48550/arXiv.1406.2661

    Args:
        `generator`: Generator model.
        `discriminator`: Discriminator model.
        `latent_dim`: Dimension of latent space, leave `None` to be parsed from
            generator. Defaults to `None`.
        `image_dim`: Dimension of synthetic image, leave `None` to be parsed from
            generator. Defaults to `None`.
    """    
    def __init__(self,
                 generator:torch.nn.Module,
                 critic:torch.nn.Module,
                 optimizer_crit:torch.optim.Optimizer,
                 optimizer_gen:torch.optim.Optimizer,
                 loss_fn:Callable[[Any], torch.Tensor]=torch.nn.BCELoss(),
                 latent_dim:Optional[int]=None,
                 image_dim:Optional[List[int]]=None,
                 device:Optional[str]=None,
                 ):
        # Parse device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.generator = generator.to(self.device)
        self.critic = critic.to(self.device)
        self.optimizer_crit = optimizer_crit
        self.optimizer_gen = optimizer_gen
        self.loss_fn = loss_fn

        if latent_dim is None:
            self.latent_dim:int = self.generator.latent_dim
        elif latent_dim is not None:
            self.latent_dim = latent_dim

        if image_dim is None:
            self.image_dim:int = self.generator.image_dim
        elif image_dim is not None:
            self.image_dim = image_dim

        # Metrics
        self.train_metrics:Dict[str, Metric] = {
            'loss_real': Mean(),
            'loss_synth': Mean(),
            'loss_gen': Mean(),
        }
        self.val_metrics:Dict[str, BinaryAccuracy] = {
            'acc_real': BinaryAccuracy(),
            'acc_synth': BinaryAccuracy(),
        }
        # To be initialize by callback History
        self.history:History
        # To be initialize by callback ProgressBar
        self.training_progress:tqdm = None
        self.train_phase_progress:tqdm = None
        self.val_phase_progress:tqdm = None

    def train_batch(self, data:Tuple[torch.Tensor, torch.Tensor]):
        # Unpack data
        x_real, _ = data
        x_real = x_real.to(self.device)
        batch_size = x_real.shape[0]
        y_synth = torch.zeros(size=(batch_size, 1))
        y_real = torch.ones(size=(batch_size, 1))

        # Phase 1 - Training the critic (discriminator)
        self.generator.eval()
        self.critic.train()
        self.optimizer_crit.zero_grad()
        ## Forward
        x_synth = self.synthesize_images(batch_size)
        pred_real = self.critic(x_real)
        pred_synth = self.critic(x_synth)
        loss_real = self.loss_fn(pred_real, y_real)
        loss_synth = self.loss_fn(pred_synth, y_synth)
        ## Back-propagation
        loss_real.backward()
        loss_synth.backward()
        self.optimizer_crit.step()

        # Phase 2 - Training the generator
        self.generator.train()
        self.critic.eval()
        self.optimizer_gen.zero_grad()
        ## Forward
        x_synth = self.synthesize_images(batch_size)
        pred_synth = self.critic(x_synth)
        loss_gen = self.loss_fn(pred_synth, y_real)
        # Back-propagation
        loss_gen.backward()
        self.optimizer_gen.step()

        with torch.inference_mode():
            # Metrics
            self.train_metrics['loss_real'].update(new_entry=loss_real)
            self.train_metrics['loss_synth'].update(new_entry=loss_synth)
            self.train_metrics['loss_gen'].update(new_entry=loss_gen)

    def test_batch(self, data:Tuple[torch.Tensor, torch.Tensor]):
        # Unpack data
        x_real, _ = data
        x_real = x_real.to(self.device)
        batch_size:int = x_real.shape[0]
        y_synth = torch.zeros(size=(batch_size, 1))
        y_real = torch.ones(size=(batch_size, 1))

        self.generator.eval()
        self.critic.eval()
        with torch.inference_mode():
            # Test 1 - Discriminator's performance on real images
            pred_real = self.critic(x_real)
            # Test 2 - Discriminator's performance on synthetic images
            x_synth = self.synthesize_images(batch_size)
            pred_synth = self.critic(x_synth)
            # Metrics
            self.val_metrics['acc_real'].update(label=y_real, prediction=pred_real)
            self.val_metrics['acc_synth'].update(label=y_synth, prediction=pred_synth)

    def synthesize_images(self, batch_size):
        latent_noise = torch.normal(mean=0, std=1, size=[batch_size, self.latent_dim])
        x_synth = self.generator(latent_noise)
        return x_synth

if __name__ == '__main__':
    from torchinfo import summary
    from dataloader import get_dataloader
    from callbacks.Callbacks import CSVLogger
    from models.GANs.utils import MakeSyntheticGIFCallback
    
    def test_mnist():
        LATENT_DIM = 100
        IMAGE_DIM = [1, 28, 28]
        NUM_EPOCHS = 20

        dataloader = get_dataloader(
            dataset='MNIST',
            resize=IMAGE_DIM[1:],
            rescale=[-1, 1],
        )

        gen = Generator(latent_dim=LATENT_DIM, image_dim=IMAGE_DIM)
        crit = Discriminator(image_dim=IMAGE_DIM, return_logits=False)
        optimizer_gen = torch.optim.Adam(params=gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
        optimizer_crit = torch.optim.Adam(params=crit.parameters(), lr=2e-4, betas=(0.5, 0.999))
        loss_fn = torch.nn.BCELoss()

        summary(model=gen, input_size=[128, LATENT_DIM])
        summary(model=crit, input_size=[128, *IMAGE_DIM])
        
        gan = GAN(
            generator=gen,
            critic=crit,
            optimizer_gen=optimizer_gen,
            optimizer_crit=optimizer_crit,
            loss_fn=loss_fn)

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