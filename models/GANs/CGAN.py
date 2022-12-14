# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from typing import Sequence, Union, Optional, Tuple

import torch
import numpy as np

from models.GANs.GAN import GAN
from models.GANs.utils import Repeat2d
from utils.modules import Reshape, OneHotEncoding, Concatenate

class ConditionalGenerator(torch.nn.Module):
    def __init__(self,
                 latent_dim:int=100,
                 image_dim:Sequence[int]=[1, 28, 28],
                 base_dim:Sequence[int]=[256, 7, 7],
                 embed_dim:Optional[int]=None,
                 num_classes:int=10,
                 onehot_input:bool=True,
                 ):
        """Initialize generator.
        
        Args:
            `latent_dim`: Dimension of latent space. Defaults to `100`.
            `image_dim`: Dimension of synthetic images. Defaults to `[28, 28, 1]`.
            `base_dim`: Dimension of the shallowest feature maps. After each
                convolutional layer, each dimension is doubled the and number of filters
                is halved until `image_dim` is reached. Defaults to `[7, 7, 256]`.
            `embed_dim`: Dimension of embedding layer. Defaults to `50`.
            `num_classes`: Number of classes. Defaults to `10`.
            `onehot_input`: `onehot_input`: Flag to indicate whether the model receives
                one-hot or label encoded target classes. Defaults to `True`.
        """
        assert isinstance(onehot_input, bool), '`onehot_input` must be of type `bool`.'
        assert isinstance(embed_dim, Optional[int]), '`embed_dim` must be of type `int` or `None`.'

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
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.onehot_input = onehot_input

        # Traditional latent branch
        self.latent_branch = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.latent_dim, out_features=np.prod(self.base_dim)),
            Reshape(out_shape=self.base_dim),
            torch.nn.ReLU(),
        )

        # Conditional label branch
        label_branch = []
        if self.onehot_input is False:
            label_branch.append(OneHotEncoding(num_classes=self.num_classes))
        if self.embed_dim is not None:
            # Replace Embedding with Dense for to accept interpolated label inputs
            label_branch.extend([
                torch.nn.Linear(in_features=self.num_classes, out_features=self.embed_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=self.embed_dim, out_features=np.prod(self.base_dim[1:])),
            ])
        else:
            label_branch.append(
                torch.nn.Linear(in_features=self.num_classes, out_features=np.prod(self.base_dim[1:])),
            )
        label_branch.extend([
            Reshape(out_shape=(1, *self.base_dim[1:])),
            torch.nn.ReLU(),
        ])
        self.label_branch = torch.nn.Sequential(*label_branch)

        # Main branch: concat both branches and upsample
        self.concat = Concatenate()
        convt_blocks = [None for i in range(num_conv)]
        for i in range(num_conv):
            in_channels = self.base_dim[0] // 2**i
            if i == 0:
                in_channels += 1
            out_channels = self.base_dim[0] // 2**(i+1)
            if i < num_conv - 1:
                convt_blocks[i] = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    torch.nn.BatchNorm2d(num_features=out_channels),
                    torch.nn.ReLU(),
                )
            elif i == num_conv - 1:
                # Last Conv2dTranspose: not use BatchNorm, replace relu with tanh
                convt_blocks[i] = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=self.image_dim[0], kernel_size=4, stride=2, padding=1, bias=False),
                    torch.nn.Tanh()
                )
        self.convt_blocks = torch.nn.Sequential(*convt_blocks)

    def forward(self, z, c):
        z = self.latent_branch(z)
        c = self.label_branch(c)
        x = self.concat([z, c])
        x = self.convt_blocks(x)
        return x

class ConditionalDiscriminator(torch.nn.Module):
    def __init__(self,
                 image_dim:Sequence[int]=[1, 28, 28],
                 base_dim:Sequence[int]=[256, 7, 7],
                 embed_dim:Optional[int]=None,
                 num_classes:int=10,
                 onehot_input:bool=True,
                 return_logits:bool=True,
                 ):
        """Initialize discriminator.
        
        Args:
            `image_dim`: Dimension of input image. Defaults to `[1, 28, 28]`.
            `base_dim`: Dimension of the shallowest feature maps, ideally equal to the
                generator's. Opposite to the generator, after each convolutional layer,
                each dimension from `image_dim` is halved and the number of filters is
                doubled until `base_dim` is reached. Defaults to `[256, 7, 7]`.
            `embed_dim`: Dimension of embedding layer. Defaults to `None`.
            `num_classes`: Number of classes. Defaults to `10`.
            `return_logits`: flag to choose between return logits or probability.
                Defaults to `True`.
        """
        assert isinstance(onehot_input, bool), '`onehot_input` must be of type bool.'
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'

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
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.onehot_input = onehot_input
        self.return_logits = return_logits

        # Conditional label branch
        label_branch = []
        if self.onehot_input is False:
            label_branch.append(OneHotEncoding(num_classes=self.num_classes))
        if self.embed_dim is not None:
            # Replace Embedding with Dense for to accept interpolated label inputs
            label_branch.extend([
                torch.nn.Linear(in_features=self.num_classes, out_features=self.embed_dim),
                torch.nn.LeakyReLU(negative_slope=0.2),
                torch.nn.Linear(in_features=self.embed_dim, out_features=np.prod(self.image_dim[1:])),
            ])
        else:
            label_branch.append(
                torch.nn.Linear(in_features=self.num_classes, out_features=np.prod(self.image_dim[1:])),
            )
        label_branch.extend([
            Reshape(out_shape=(1, *self.image_dim[1:])),
            torch.nn.LeakyReLU(negative_slope=0.2),
        ])
        self.label_branch = torch.nn.Sequential(*label_branch)

        # Main branch: concat both branches and downsample
        self.concat = Concatenate()
        conv_blocks = [None for i in range(num_conv)]
        for i in range(num_conv):
            in_channels = self.base_dim[0] // 2**(num_conv-i)
            out_channels = self.base_dim[0] // 2**(num_conv-1-i)
            if i == 0:
                # First Conv2d: not use BatchNorm 
                in_channels = self.image_dim[0] + 1
                conv_blocks[i] = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    torch.nn.LeakyReLU(negative_slope=0.2),
                )
            elif i > 0:
                conv_blocks[i] = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    torch.nn.BatchNorm2d(num_features=out_channels),
                    torch.nn.LeakyReLU(negative_slope=0.2),
                )
        self.conv_blocks = torch.nn.Sequential(*conv_blocks)
        
        self.flatten = torch.nn.Flatten()
        self.logits = torch.nn.Linear(in_features=np.prod(self.base_dim), out_features=1)
        if self.return_logits is False:
            self.pred = torch.nn.Sigmoid()

    def forward(self, img, c):
        c = self.label_branch(c)
        x = self.concat([img, c])
        x = self.conv_blocks(x)
        x = self.flatten(x)
        x = self.logits(x)
        if self.return_logits is False:
            x = self.pred(x)
        return x

class ConditionalGeneratorStack(torch.nn.Module):
    def __init__(self,
                 latent_dim:int=100,
                 image_dim:Sequence[int]=[1, 28, 28],
                 base_dim:Sequence[int]=[256, 7, 7],
                 num_classes:int=10,
                 onehot_input:bool=True,
                 ):
        assert isinstance(onehot_input, bool), '`onehot_input` must be of type `bool`.'

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
        self.num_classes = num_classes
        self.onehot_input = onehot_input

        if self.onehot_input is False:
            self.oh_encode = OneHotEncoding(num_classes=self.num_classes)

        # Main branch: concat both branches and upsample
        self.concat = Concatenate()
        
        self.dense_0 = torch.nn.Linear(in_features=self.latent_dim + self.num_classes, out_features=np.prod(self.base_dim[1:])*self.base_dim[0])
        self.lrelu_0 = torch.nn.LeakyReLU(negative_slope=0.2)
        self.reshape = Reshape(out_shape=self.base_dim)

        convt_blocks = [None for i in range(num_conv)]
        for i in range(num_conv):
            in_channels = self.base_dim[0] // 2**i
            out_channels = self.base_dim[0] // 2**(i+1)
            if i < num_conv - 1:
                convt_blocks[i] = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    torch.nn.BatchNorm2d(num_features=out_channels),
                    torch.nn.ReLU(),
                )
            elif i == num_conv - 1:
                # Last Conv2dTranspose: not use BatchNorm, replace relu with tanh
                convt_blocks[i] = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=self.image_dim[0], kernel_size=4, stride=2, padding=1, bias=False),
                    torch.nn.Tanh()
                )
        self.convt_blocks = torch.nn.Sequential(*convt_blocks)

    def forward(self, z, c):
        c = self.oh_encode(c)
        x = self.concat([z, c])
        x = self.dense_0(x)
        x = self.lrelu_0(x)
        x = self.reshape(x)
        x = self.convt_blocks(x)
        return x
    
class ConditionalDiscriminatorStack(torch.nn.Module):
    def __init__(self,
                 image_dim:Sequence[int]=[1, 28, 28],
                 base_dim:Sequence[int]=[256, 7, 7],
                 num_classes:int=10,
                 onehot_input:bool=True,
                 return_logits:bool=True,
                 ):
        assert isinstance(onehot_input, bool), '`onehot_input` must be of type bool.'
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'

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
        self.num_classes = num_classes
        self.onehot_input = onehot_input
        self.return_logits = return_logits

        # Conditional label branch
        label_branch = []
        if self.onehot_input is False:
            label_branch.append(OneHotEncoding(num_classes=self.num_classes),)
        label_branch.append(Repeat2d(repeats=self.image_dim[1:]))
        self.label_branch = torch.nn.Sequential(*label_branch)

        # Main branch: concat both branches and downsample
        self.concat = Concatenate()
        conv_blocks = [None for i in range(num_conv)]
        for i in range(num_conv):
            in_channels = self.base_dim[0] // 2**(num_conv-i)
            out_channels = self.base_dim[0] // 2**(num_conv-1-i)
            if i == 0:
                # First Conv2d: not use BatchNorm 
                in_channels = self.image_dim[0] + self.num_classes
                conv_blocks[i] = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    torch.nn.LeakyReLU(negative_slope=0.2),
                )
            elif i > 0:
                conv_blocks[i] = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    torch.nn.BatchNorm2d(num_features=out_channels),
                    torch.nn.LeakyReLU(negative_slope=0.2),
                )
        self.conv_blocks = torch.nn.Sequential(*conv_blocks)

        self.flatten = torch.nn.Flatten()
        self.logits = torch.nn.Linear(in_features=np.prod(self.base_dim), out_features=1)
        if self.return_logits is False:
            self.pred = torch.nn.Sigmoid()

    def forward(self, img, c):
        c = self.label_branch(c)
        x = self.concat([img, c])
        x = self.conv_blocks(x)
        x = self.flatten(x)
        x = self.logits(x)
        if self.return_logits is False:
            x = self.pred(x)
        return x

class CGAN(GAN):
    """Conditional Generative Adversarial Network.
    
    Args:
        `generator`: Generator model. Defaults to `Generator()`.
        `discriminator`: Discriminator model. Defaults to `Discriminator()`.
        `latent_dim`: Dimension of latent space, leave as `None` to be parsed from
            generator. Defaults to `None`.
        `image_dim`: Dimension of synthetic image, leave as `None` to be parsed from
            generator. Defaults to `None`.
        `embed_dim`: Dimension of embedding vector, leave as `None` to be parsed
            from generator. Defaults to `None`.
        `num_classes`: Number of classes, leave as `None` to be parsed from
            generator. Defaults to `None`.
    """    
    def __init__(self,
                 generator:torch.nn.Module,
                 critic:torch.nn.Module,
                 latent_dim:Optional[int]=None,
                 image_dim:Optional[Sequence[int]]=None,
                 num_classes:Union[None, int]=None,
                 onehot_input:Union[None, bool]=None,
                 device:Optional[str]=None):
        """Initialize cGAN.
        
        Args:
            `generator`: Generator model. Defaults to `Generator()`.
            `discriminator`: Discriminator model. Defaults to `Discriminator()`.
            `latent_dim`: Dimension of latent space, leave as `None` to be parsed from
                generator. Defaults to `None`.
            `image_dim`: Dimension of synthetic image, leave as `None` to be parsed from
                generator. Defaults to `None`.
            `num_classes`: Number of classes, leave as `None` to be parsed from
                generator. Defaults to `None`.
        """
        super().__init__(
            generator=generator,
            critic=critic,
            latent_dim=latent_dim,
            image_dim=image_dim,
            device=device,
        )

        if num_classes is None:
            self.num_classes:int = self.generator.num_classes
        elif num_classes is not None:
            self.num_classes = num_classes

        if onehot_input is None:
            self.onehot_input:bool = self.generator.onehot_input
        elif onehot_input is not None:
            self.onehot_input = onehot_input

    def train_batch(self, data:Tuple[torch.Tensor, torch.Tensor]):
        '''
        Notation:
            label: correspoding to label in training set (0 to `num_classes - 1`)
            x: image (synthetic or real)
            y/pred: validity/prediction of image (0 for synthetic, 1 for real)
        '''
        # Unpack data
        x_real, label = data
        x_real = x_real.to(self.device)
        label = label.to(self.device)
        batch_size = x_real.shape[0]
        y_synth = torch.zeros(size=(batch_size, 1))
        y_real = torch.ones(size=(batch_size, 1))

        # Phase 1 - Training the discriminator
        self.generator.eval()
        self.critic.train()
        self.optimizer_crit.zero_grad()
        # Forward
        x_synth = self.synthesize_images(label, batch_size)
        pred_real = self.critic(x_real, label)
        pred_synth = self.critic(x_synth, label)
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
        x_synth = self.synthesize_images(label, batch_size)
        pred_synth = self.critic(x_synth, label)
        loss_gen = self.loss_fn(pred_synth, y_real)
        # Back-propagation
        loss_gen.backward()
        self.optimizer_gen.step()

        with torch.inference_mode():
            # Metrics
            self.train_metrics['loss_real'].update(new_entry=loss_real)
            self.train_metrics['loss_synth'].update(new_entry=loss_synth)
            self.train_metrics['loss_gen'].update(new_entry=loss_gen)

    def test_batch(self, data):
        # Unpack data
        x_real, label = data
        x_real = x_real.to(self.device)
        label = label.to(self.device)
        batch_size = x_real.shape[0]
        y_synth = torch.zeros(size=(batch_size, 1))
        y_real = torch.ones(size=(batch_size, 1))

        self.generator.eval()
        self.critic.eval()
        with torch.inference_mode():
            # Test 1 - Discriminator performs on real data
            pred_real = self.critic(x_real, label)
            # Test 2 - Generator tries to fool discriminator
            x_synth = self.synthesize_images(label, batch_size)
            pred_synth = self.critic(label, x_synth)
            # Metrics
            self.val_metrics['acc_real'].update(label=y_real, prediction=pred_real)
            self.val_metrics['acc_synth'].update(label=y_synth, prediction=pred_synth)

    def synthesize_images(self, label, batch_size):
        latent_noise = torch.normal(mean=0, std=1, size=[batch_size, self.latent_dim])
        x_synth = self.generator(latent_noise, label)
        return x_synth

if __name__ == '__main__':
    from torchinfo import summary

    from utils.dataloader import get_dataloader
    # from models.distillers.CDAFL import ConditionalDataFreeGenerator, ConditionalLenet5_ReLU_MaxPool, ConditionalResNet_DAFL
    from utils.callbacks import CSVLogger
    from models.GANs.utils import MakeConditionalSyntheticGIFCallback, MakeInterpolateSyntheticGIFCallback

    def test_mnist():
        LATENT_DIM = 100
        IMAGE_DIM = [1, 28, 28]
        BASE_DIM = [256, 7, 7]
        EMBED_DIM = 50
        NUM_CLASSES = 10
        BATCH_SIZE = 128
        NUM_EPOCHS = 20

        dataloader = get_dataloader(
            dataset='MNIST',
            resize=IMAGE_DIM[1:],
            rescale=[-1, 1],
            batch_size_train=BATCH_SIZE,
            onehot_label=True
        )

        gen = ConditionalGenerator(
            latent_dim=LATENT_DIM,
            image_dim=IMAGE_DIM,
            base_dim=BASE_DIM,
            embed_dim=EMBED_DIM,
            num_classes=NUM_CLASSES,
            onehot_input=True
        )
        crit = ConditionalDiscriminator(
            image_dim=IMAGE_DIM,
            base_dim=BASE_DIM,
            embed_dim=EMBED_DIM,
            num_classes=NUM_CLASSES,
            onehot_input=True,
            return_logits=False,
        )
        
        summary(model=gen, input_size=[[BATCH_SIZE, LATENT_DIM], [BATCH_SIZE, NUM_CLASSES]])
        summary(model=crit, input_size=[[BATCH_SIZE, *IMAGE_DIM], [BATCH_SIZE, NUM_CLASSES]])
        
        gan = CGAN(generator=gen, critic=crit)
        gan.compile(
            optimizer_gen=torch.optim.Adam(params=gen.parameters(), lr=2e-4, betas=(0.5, 0.999)),
            optimizer_crit=torch.optim.Adam(params=crit.parameters(), lr=2e-4, betas=(0.5, 0.999)),
            loss_fn=torch.nn.BCELoss())

        csv_logger = CSVLogger(
            filename=f'./logs/{gan.__class__.__name__}.csv',
            append=True
        )
        gif_maker = MakeConditionalSyntheticGIFCallback(
            filename=f'./logs/{gan.__class__.__name__}.gif',
            postprocess_fn=lambda x:(x+1)/2,
            class_names=dataloader['train'].dataset.classes

        )
        slerper = MakeInterpolateSyntheticGIFCallback(
            filename=f'./logs/{gan.__class__.__name__}.gif',
            itpl_method='slerp',
            postprocess_fn=lambda x:(x+1)/2,
            class_names=dataloader['train'].dataset.classes

        )
        gan.training_loop(
            trainloader=dataloader['train'],
            num_epochs=NUM_EPOCHS,
            valloader=dataloader['val'],
            callbacks=[csv_logger, gif_maker, slerper],
        )
    
    test_mnist()

    def experiment_mnist_CGAN():
        LATENT_DIM = 100
        IMAGE_DIM = [28, 28, 1]
        BASE_DIM = [7, 7, 256]
        NUM_CLASSES = 10
        BATCH_SIZE = 256

        OPTIMIZER_GEN = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
        OPTIMIZER_DISC = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)

        ds, ds_info = dataloader(
            dataset='mnist',
            rescale=[-1, 1],
            batch_size_train=BATCH_SIZE,
            batch_size_test=1000,
            drop_remainder=True,
            onehot_label=True,
            with_info=True
        )
        class_names = ds_info.features['label'].names

        gen = ConditionalGeneratorEmbed(
            latent_dim=LATENT_DIM,
            image_dim=IMAGE_DIM,
            base_dim=BASE_DIM,
            onehot_input=True
        )
        gen.build()

        disc = ConditionalDiscriminatorEmbed(
            image_dim=IMAGE_DIM,
            base_dim=BASE_DIM,
            num_classes=NUM_CLASSES,
            onehot_input=True
        )
        disc.build()
        disc.compile(metrics=['accuracy'])

        gan = CGAN(generator=gen, discriminator=disc)
        gan.build()
        gan.summary(with_graph=True, expand_nested=True, line_length=120)

        gan.compile(
            optimizer_gen=OPTIMIZER_GEN,
            optimizer_disc=OPTIMIZER_DISC,
            loss_fn = keras.losses.BinaryCrossentropy()
        )
        csv_logger = keras.callbacks.CSVLogger(
            filename=f'./logs/{gan.name}_{gan.generator.name}_{gan.discriminator.name}.csv',
            append=True
        )
        gif_maker = MakeConditionalSyntheticGIFCallback(
            filename=f'./logs/{gan.name}_{gan.generator.name}_{gan.discriminator.name}.gif',
            postprocess_fn=lambda x:x*0.5 + 0.5,
            normalize=False,
            class_names=class_names
        )
        slerper = MakeInterpolateSyntheticGIFCallback(
            filename=f'./logs/{gan.name}_{gan.generator.name}_{gan.discriminator.name}_itpl_slerp.gif',
            itpl_method='slerp',
            postprocess_fn=lambda x:(x+1)/2,
            class_names=class_names
        )
        gan.fit(
            x=ds['train'],
            epochs=50,
            callbacks=[csv_logger, gif_maker, slerper],
            validation_data=ds['test']
        )

    def experiment_mnist_CGAN_with_DAFL_models(embed_dim:Union[int, None]=None):
        LATENT_DIM = 100
        IMAGE_DIM = [32, 32, 1]
        EMBED_DIM = embed_dim
        NUM_CLASSES = 10
        BATCH_SIZE = 256
        NUM_EPOCHS = 50

        OPTIMIZER_GEN = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        OPTIMIZER_DISC = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        
        ds, info = dataloader(
            dataset='mnist',
            resize=IMAGE_DIM[0:-1],
            rescale='standardization',
            batch_size_train=BATCH_SIZE,
            batch_size_test=1000,
            drop_remainder=False,
            onehot_label=True,
            with_info=True)
        class_names = info.features['label'].names

        gen = ConditionalDataFreeGenerator(
            latent_dim=LATENT_DIM,
            image_dim=IMAGE_DIM,
            embed_dim=EMBED_DIM,
            num_classes=NUM_CLASSES,
            onehot_input=True,
            dafl_batchnorm=True
        )
        gen.build()

        disc = ConditionalLenet5_ReLU_MaxPool(
            input_dim=IMAGE_DIM,
            embed_dim=EMBED_DIM,
            num_classes=NUM_CLASSES,
            onehot_input=True,
        )
        disc.build()

        gan = CGAN(generator=gen, discriminator=disc)
        gan.build()
        gan.summary(with_graph=True, expand_nested=True, line_length=120)
        gan.compile(
            optimizer_gen=OPTIMIZER_GEN,
            optimizer_disc=OPTIMIZER_DISC,
            loss_fn=keras.losses.BinaryCrossentropy(),
        )

        csv_logger = keras.callbacks.CSVLogger(
            f'./logs/{gan.name}_{gan.generator.name}_{gan.discriminator.name}.csv',
            append=True)
        
        gif_maker = MakeConditionalSyntheticGIFCallback(
            filename=f'./logs/{gan.name}_{gan.generator.name}_{gan.discriminator.name}.gif', 
            postprocess_fn=lambda x:x*0.3081 + 0.1307,
            normalize=False,
            class_names=class_names,
            delete_png=False
        )
        gan.fit(
            x=ds['train'],
            epochs=NUM_EPOCHS,
            callbacks=[csv_logger, gif_maker],
            validation_data=ds['test'],
        )

    def experiment_cifar10_CGAN_with_DAFL_models(ver:int=34, embed_dim:Union[int, None]=None):
        VER = ver
        LATENT_DIM = 1000
        IMAGE_DIM = [32, 32, 3]
        EMBED_DIM = embed_dim
        NUM_CLASSES = 10
        BATCH_SIZE = 128
        NUM_EPOCHS = 200

        OPTIMIZER_GEN = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        OPTIMIZER_DISC = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        
        def augmentation_fn(x):
            x = tf.pad(tensor=x, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]], mode='SYMMETRIC')
            x = tf.image.random_crop(value=x, size=[tf.shape(x)[0], *IMAGE_DIM])
            x = tf.image.random_flip_left_right(image=x)
            return x
        ds, info = dataloader(
            dataset='cifar10',
            augmentation_fn=augmentation_fn,
            rescale='standardization',
            batch_size_train=BATCH_SIZE,
            batch_size_test=1000,
            drop_remainder=True,
            onehot_label=True,
            with_info=True,
        )
        class_names = info.features['label'].names

        gen = ConditionalDataFreeGenerator(
            latent_dim=LATENT_DIM,
            image_dim=IMAGE_DIM,
            embed_dim=EMBED_DIM,
            num_classes=NUM_CLASSES,
            onehot_input=True,
            dafl_batchnorm=True
        )
        gen.build()

        disc = ConditionalResNet_DAFL(
            ver=VER,
            input_dim=IMAGE_DIM,
            embed_dim=EMBED_DIM,
            num_classes=NUM_CLASSES,
            onehot_input=True,
        )
        disc.build()

        gan = CGAN(generator=gen, discriminator=disc)
        gan.build()
        gan.summary(with_graph=True, expand_nested=True, line_length=120)
        gan.compile(
            optimizer_gen=OPTIMIZER_GEN,
            optimizer_disc=OPTIMIZER_DISC,
            loss_fn=keras.losses.BinaryCrossentropy(),
        )

        csv_logger = keras.callbacks.CSVLogger(
            f'./logs/{gan.name}_{gan.generator.name}_{gan.discriminator.name}.csv',
            append=True)
        
        gif_maker = MakeConditionalSyntheticGIFCallback(
            filename=f'./logs/{gan.name}_{gan.generator.name}_{gan.discriminator.name}.gif', 
            postprocess_fn=lambda x:x*tf.constant([[[0.2470, 0.2435, 0.2616]]]) + tf.constant([[[0.4914, 0.4822, 0.4465]]]),
            normalize=False,
            class_names=class_names,
            delete_png=False
        )
        gan.fit(
            x=ds['train'],
            epochs=NUM_EPOCHS,
            callbacks=[csv_logger, gif_maker],
            validation_data=ds['test'],
        )

    experiment_cifar10_CGAN_with_DAFL_models()