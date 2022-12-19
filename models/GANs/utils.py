# Change path
import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

import glob
from typing import Callable, Union, Any, Sequence, Literal
import warnings

import PIL
import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from utils.callbacks import Callback

class Repeat2d(torch.nn.Module):
    """Repeats the input based on given size.

    Typically, it is used for the discriminator in conditional GAN; spefically to
    repeat a multi-hot/one-hot vector to a stack of all-ones and and all-zeros
    images (before concatenating with real images).

    Args:
        `repeats`: Number of times to repeat the width and height.
    """
    NUM_REPEATS = 2
    def __init__(self, repeats:Sequence[int], **kwargs):
        """Initialize layer.
        
        Args:
            `repeats`: Number of times to repeat the width and height.
        """
        if any([not isinstance(item, int) for item in repeats]):
            raise TypeError(
                f"Expected a sequence of {self.NUM_REPEATS} integers, got {type(repeats)}."
            )
        if len(repeats) != self.NUM_REPEATS:
            raise ValueError(
                f"Expected a sequence of {self.NUM_REPEATS} integers, got {type(repeats)}."
            )
        super().__init__(**kwargs)
        self.repeats = repeats

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        for dim in torch.arange(start=2, end=2+self.NUM_REPEATS):
            x = x.unsqueeze(dim=dim)
        return x.repeat(repeats=[1, 1, *self.repeats])

class MakeSyntheticGIFCallback(Callback):
    """Callback to generate synthetic images, typically used with a Generative
    Adversarial Network.
    
    Args:
        `filename`: Path to save GIF to. Defaults to `'./logs/GAN.gif'`.
        `nrows`: Number of rows in subplot figure. Defaults to `5`.
        `ncols`: Number of columns in subplot figure. Defaults to `5`.
        `postprocess_fn`: Post-processing function to map synthetic images back to
            the plot range, ideally [0, 1]. Leave as `None` to skip post-processing.
            Defaults to `None`.
        `latent_dim`: Dimension of latent space, leave as `None` to be parsed from
            model. Defaults to `None`.
        `image_dim`: Dimension of synthetic images, leave as `None` to be parsed
            from model. Defaults to `None`.
        `keep_noise`: Flag to feed the same latent noise to generator for the whole
            training. Defaults to `True`.
        `delete_png`: Flag to delete PNG files and folder at `filename/png` after
            training. Defaults to `True`.
        `duration`: Duration of the generated GIF in milliseconds.
            Defaults to `5000`.
    """
    def __init__(self,
                 filename:str='./logs/GAN.gif',
                 nrows:int=5,
                 ncols:int=5,
                 postprocess_fn:Union[None, Callable[[Any], Any]]=None,
                 normalize:bool=True,
                 latent_dim:Union[None, int]=None,
                 image_dim:Union[None, Sequence[int]]=None,
                 keep_noise:bool=True,
                 seed:Union[None, int]=None,
                 delete_png:bool=True,
                 save_freq:int=1,
                 duration:float=5000):
        """Initialize callback.
        
        Args:
            `filename`: Path to save GIF to. Defaults to `'./logs/GAN.gif'`.
            `nrows`: Number of rows in subplot figure. Defaults to `5`.
            `ncols`: Number of columns in subplot figure. Defaults to `5`.
            `postprocess_fn`: Post-processing function to map synthetic images back to
                the plot range, ideally [0, 1]. Leave as `None` to skip post-processing.
                Defaults to `None`.
            `latent_dim`: Dimension of latent space, leave as `None` to be parsed from
                model. Defaults to `None`.
            `image_dim`: Dimension of synthetic images, leave as `None` to be parsed
                from model. Defaults to `None`.
            `keep_noise`: Flag to feed the same latent noise to generator for the whole
                training. Defaults to `True`.
            `delete_png`: Flag to delete PNG files and folder at `filename/png` after
                training. Defaults to `True`.
            `duration`: Duration of the generated GIF in milliseconds.
                Defaults to `5000`.
        """
        super().__init__()
        self.filename = filename
        self.nrows = nrows
        self.ncols = ncols
        self.postprocess_fn = postprocess_fn
        self.normalize = normalize
        self.latent_dim = latent_dim
        self.image_dim = image_dim
        self.keep_noise = keep_noise
        self.seed = seed
        self.delete_png = delete_png
        self.save_freq = save_freq
        self.duration = duration

        self.path_png_folder = self.filename[0:-4] + '_png'

    def on_train_begin(self, logs=None):
        self.handle_args()
        # Renew/create folder containing PNG files
        if os.path.isdir(self.path_png_folder):
            for png in glob.glob(f'{self.path_png_folder}/*.png'):
                os.remove(png)
        else:
            os.mkdir(self.path_png_folder)
        self.precompute_inputs()

    def on_epoch_end(self, epoch, logs=None):
        x_synth = self.synthesize_images()
        if epoch % self.save_freq == 0:
            self.make_figure(x_synth.clone().detach().to('cpu'), epoch)

    def on_train_end(self, logs=None):
        # Make GIF
        path_png = f'{self.path_png_folder}/*.png'
        img, *imgs = [PIL.Image.open(f) for f in sorted(glob.glob(path_png))]
        img.save(
            fp=self.filename,
            format='GIF',
            append_images=imgs,
            save_all=True,
            duration=self.duration/(len(imgs) + 1),
            loop=0)

        if self.delete_png is True:
            for png in glob.glob(path_png):
                os.remove(png)
            os.rmdir(self.path_png_folder)

    def handle_args(self):
        """Handle input arguments to callback, as some are not accessible in __init__().
        """
        if self.postprocess_fn is None:
            self.postprocess_fn = lambda x:x

        if self.normalize is True:
            self.vmin, self.vmax = None, None
        elif self.normalize is False:
            self.vmin, self.vmax = 0, 1

        if self.latent_dim is None:
            self.latent_dim:int = self.host.latent_dim

        if self.image_dim is None:
            self.image_dim:int = self.host.image_dim

    def precompute_inputs(self):
        """Pre-compute inputs to feed to the generator. Eg: latent noise.
        """
        batch_size = self.nrows*self.ncols
        if self.seed is None:
            self.latent_noise = torch.normal(mean=0, std=1, size=(batch_size, self.latent_dim), device=self.device)
        elif self.seed is not None:
            with torch.manual_seed(self.seed):
                self.latent_noise = torch.normal(mean=0, std=1, size=(batch_size, self.latent_dim), device=self.device)

    def synthesize_images(self) -> torch.Tensor:
        """Produce synthetic images with the generator.
        
        Returns:
            A batch of synthetic images.
        """
        if self.keep_noise is False:
            batch_size = self.nrows*self.ncols
            if self.seed is None:
                self.latent_noise = torch.normal(mean=0, std=1, size=(batch_size, self.latent_dim), device=self.device)
            elif self.seed is not None:
                with torch.manual_seed(self.seed):
                    self.latent_noise = torch.normal(mean=0, std=1, size=(batch_size, self.latent_dim), device=self.device)
        
        self.host.generator.eval()
        with torch.inference_mode():
            x_synth = self.host.generator(self.latent_noise)
            x_synth = self.postprocess_fn(x_synth)
        return x_synth

    def make_figure(self, x_synth:torch.Tensor, value:Union[int, float]):
        """Tile the synthetic images into a nice grid, then make and save a figure at
        the given epoch.
        
        Args:
            `x_synth`: A batch of synthetic images.
            `epoch`: Current epoch.
        """
        fig, ax = plt.subplots(constrained_layout=True, figsize=(self.ncols, 0.5 + self.nrows))
        self.modify_suptitle(figure=fig, value=value)

        # Tile images into a grid
        x = x_synth
        # Pad 1 pixel on top row & left column to all images in batch
        x = torch.nn.functional.pad(x, pad=(1, 0, 1, 0), value=1) # top, bottom, left, right
        x = torch.reshape(x, shape=[self.nrows, self.ncols, *x.shape[1:]])
        x = torch.concat(torch.unbind(x, dim=0), dim=2)
        x = torch.concat(torch.unbind(x, dim=0), dim=2)
        # Crop 1 pixel on top row & left column from the concatenated image
        x = x[:, 1:, 1:]
        x = x.permute(1, 2, 0)

        self.modify_axis(axis=ax)

        if self.image_dim[0] == 1:
            ax.imshow(x.squeeze(axis=-1), cmap='gray', vmin=self.vmin, vmax=self.vmax)
        elif self.image_dim[0] > 1:
            ax.imshow(x, vmin=self.vmin, vmax=self.vmax)

        fig.savefig(self.modify_savepath(value=value))
        plt.close(fig)

    def modify_suptitle(self, figure:Figure, value:int):
        figure.suptitle(f'{self.host.__class__.__name__} - Epoch {value}')

    def modify_axis(self, axis:Axes):
        axis.axis('off')

    def modify_savepath(self, value:int):
        return f"{self.path_png_folder}/{self.host.__class__.__name__}_epoch_{value:04d}.png"

class MakeConditionalSyntheticGIFCallback(MakeSyntheticGIFCallback):
    """Callback to generate synthetic images, typically used with a Conditional
    Generative Adversarial Network.

    Args:
        `filename`: Path to save GIF to. Defaults to `'./logs/CGAN.gif'`.
        `target_classes`: The conditional target classes to make synthetic images,
            also is the columns in the figure. Leave as `None` to include all
            classes. Defaults to `None`.
        `num_samples_per_class`: Number of sample per class, also is the number of
            rows in the figure. Defaults to `5`.
        `postprocess_fn`: Post-processing function to map synthetic images back to
            the plot range, ideally [0, 1]. Leave as `None` to skip post-processing.  
            Defaults to `None`.
        `latent_dim`: Dimension of latent space, leave as `None` to be parsed from
            model. Defaults to `None`.
        `image_dim`: Dimension of synthetic images, leave as `None` to be parsed
            from model. Defaults to `None`.
        `num_classes`: Number of classes, leave as `None` to be parsed from model.
            Defaults to `None`.
        `class_names`: Sequence of name of labels, should have length equal to total
            number of classes. Leave as `None` for generic `'class x'` names.
            Defaults to `None`.
        `onehot_label`: Flag to indicate whether the GAN model/generator receives
            one-hot or label encoded target classes, leave as `None` to be parsed
            from model. Defaults to `None`.
        `keep_noise`: Flag to feed the same latent noise to generator for the whole
            training. Defaults to `True`.
        `delete_png`: Flag to delete PNG files and folder at `filename/png` after
            training. Defaults to `True`.
        `duration`: Duration of the generated GIF in milliseconds.
            Defaults to `5000`.
    """
    def __init__(self,
                 filename:str='./logs/CGAN.gif',
                 target_classes:Union[None, Sequence[int]]=None,
                 num_samples_per_class:int=5,
                 postprocess_fn:Union[None, Callable[[Any], Any]]=None,
                 normalize:bool=True,
                 latent_dim:Union[None, int]=None,
                 image_dim:Union[None, Sequence[int]]=None,
                 num_classes:Union[None, int]=None,
                 class_names:Union[None, Sequence[str]]=None,
                 onehot_label:Union[None, bool]=None,
                 keep_noise:bool=True,
                 seed:Union[None, int]=None,
                 delete_png:bool=True,
                 save_freq:int=1,
                 duration:float=5000):
        """Initialize callback.
        
        Args:
            `filename`: Path to save GIF to. Defaults to `'./logs/CGAN.gif'`.
            `target_classes`: The conditional target classes to make synthetic images,
                also is the columns in the figure. Leave as `None` to include all
                classes. Defaults to `None`.
            `num_samples_per_class`: Number of sample per class, also is the number of
                rows in the figure. Defaults to `5`.
            `postprocess_fn`: Post-processing function to map synthetic images back to
                the plot range, ideally [0, 1]. Leave as `None` to skip post-processing.  
                Defaults to `None`.
            `latent_dim`: Dimension of latent space, leave as `None` to be parsed from
                model. Defaults to `None`.
            `image_dim`: Dimension of synthetic images, leave as `None` to be parsed
                from model. Defaults to `None`.
            `num_classes`: Number of classes, leave as `None` to be parsed from model.
                Defaults to `None`.
            `class_names`: Sequence of name of labels, should have length equal to total
                number of classes. Leave as `None` for generic `'class x'` names.
                Defaults to `None`.
            `onehot_label`: Flag to indicate whether the GAN model/generator receives
                one-hot or label encoded target classes, leave as `None` to be parsed
                from model. Defaults to `None`.
            `keep_noise`: Flag to feed the same latent noise to generator for the whole
                training. Defaults to `True`.
            `delete_png`: Flag to delete PNG files and folder at `filename/png` after
                training. Defaults to `True`.
            `duration`: Duration of the generated GIF in milliseconds.
                Defaults to `5000`.
        """                 
        super(MakeConditionalSyntheticGIFCallback, self).__init__(
            filename=filename,
            nrows=None,
            ncols=None,
            postprocess_fn=postprocess_fn,
            normalize=normalize,
            latent_dim=latent_dim,
            image_dim=image_dim,
            keep_noise=keep_noise,
            seed=seed,
            delete_png=delete_png,
            save_freq=save_freq,
            duration=duration,
        )
        self.target_classes = target_classes
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        self.class_names = class_names
        self.onehot_label = onehot_label

    def handle_args(self):
        super(MakeConditionalSyntheticGIFCallback, self).handle_args()
        if self.num_classes is None:
            self.num_classes:int = self.host.num_classes

        if self.class_names is None:
            self.class_names = [f'Class {i}' for i in range(self.num_classes)]

        if self.onehot_label is None:
            self.onehot_label:bool = self.host.onehot_label

        if self.target_classes is None:
            self.target_classes = [label for label in range(self.num_classes)]
        
        self.nrows = self.num_samples_per_class
        self.ncols = len(self.target_classes)

    def precompute_inputs(self):
        super().precompute_inputs()

        self.label = torch.tensor(self.target_classes, dtype=torch.long, device=self.device).repeat([self.nrows])
        if self.onehot_label is True:
            self.label = torch.nn.functional.one_hot(input=self.label, num_classes=self.num_classes)
        self.label = self.label.to(dtype=torch.float)

    def synthesize_images(self):
        if self.keep_noise is False:
            batch_size = self.nrows*self.ncols
            self.latent_noise = torch.normal(mean=0, std=1, size=[batch_size, self.latent_dim], device=self.device)
        
        self.host.generator.eval()
        with torch.inference_mode():
            x_synth = self.host.generator(self.latent_noise, self.label)
            x_synth = self.postprocess_fn(x_synth)
        return x_synth

    def modify_axis(self, axis:Axes):
        xticks = (self.image_dim[1] + 1)*torch.arange(len(self.target_classes)) + self.image_dim[1]/2
        xticklabels = [self.class_names[label] for label in self.target_classes]
        
        axis.set_frame_on(False)
        axis.tick_params(axis='both', length=0)
        axis.set(yticks=[], xticks=xticks, xticklabels=xticklabels)

class MakeInterpolateSyntheticGIFCallback(MakeSyntheticGIFCallback):
    """Callback to generate synthetic images, interpolated between the classes of a
    Conditional Generative Adversarial Network.
    
    The callback can only work with models receiving one-hot encoded inputs. It
    will make figures at the end of the last epoch.
    
    Args:
        `filename`: Path to save GIF to. Defaults to `'./logs/GAN_itpl.gif'`.
        `start_classes`: Classes at the start of interpolation along the rows, leave
            as `None` to include all classes. Defaults to `None`.
        `stop_classes`: Classes at the stop of interpolation along the columns, leave
            as `None` to include all classes. Defaults to `None`.
        `num_interpolate`: Number of interpolation. Defaults to `21`.
        `postprocess_fn`: Post-processing function to map synthetic images back to
            the plot range, ideally [0, 1]. Leave as `None` to skip post-processing.  
            Defaults to `None`.
        `latent_dim`: Dimension of latent space, leave as `None` to be parsed from
            model. Defaults to `None`.
        `image_dim`: Dimension of synthetic images, leave as `None` to be parsed
            from model. Defaults to `None`.
        `num_classes`: Number of classes, leave as `None` to be parsed from model.
            Defaults to `None`.
        `class_names`: Sequence of name of labels, should have length equal to total
            number of classes. Leave as `None` for generic `'class x'` names.
            Defaults to `None`.
        `keep_noise`: Flag to feed the same latent noise to generator for the whole
            training. Defaults to `True`.
        `delete_png`: Flag to delete PNG files and folder at `filename/png` after
            training. Defaults to `True`.
        `duration`: Duration of the generated GIF in milliseconds.
            Defaults to `5000`.
    """
    def __init__(self,
                 filename:str='./logs/GAN_itpl.gif',
                 start_classes:Sequence[int]=None,
                 stop_classes:Sequence[int]=None,
                 num_itpl:int=51,
                 itpl_method:Literal['linspace', 'slerp']='linspace',
                 postprocess_fn:Union[None, Callable[[Any], Any]]=None,
                 normalize:bool=True,
                 latent_dim:Union[None, int]=None,
                 image_dim:Union[None, Sequence[int]]=None,
                 num_classes:Union[None, int]=None,
                 class_names:Union[None, Sequence[str]]=None,
                 keep_noise:bool=True,
                 seed:Union[None, int]=None,
                 delete_png:bool=True,
                 duration:float=5000,
                 **kwargs):
        """Initialize callback.
        
        Args:
            `filename`: Path to save GIF to. Defaults to `'./logs/GAN_itpl.gif'`.
            `start_classes`: Classes at the start of interpolation along the rows, leave
                as `None` to include all classes. Defaults to `None`.
            `stop_classes`: Classes at the stop of interpolation along the columns, leave
                as `None` to include all classes. Defaults to `None`.
            `num_interpolate`: Number of interpolation. Defaults to `21`.
            `postprocess_fn`: Post-processing function to map synthetic images back to
                the plot range, ideally [0, 1]. Leave as `None` to skip post-processing.  
                Defaults to `None`.
            `latent_dim`: Dimension of latent space, leave as `None` to be parsed from
                model. Defaults to `None`.
            `image_dim`: Dimension of synthetic images, leave as `None` to be parsed
                from model. Defaults to `None`.
            `num_classes`: Number of classes, leave as `None` to be parsed from model.
                Defaults to `None`.
            `class_names`: Sequence of name of labels, should have length equal to total
                number of classes. Leave as `None` for generic `'class x'` names.
                Defaults to `None`.
            `keep_noise`: Flag to feed the same latent noise to generator for the whole
                training. Defaults to `True`.
            `delete_png`: Flag to delete PNG files and folder at `filename/png` after
                training. Defaults to `True`.
            `duration`: Duration of the generated GIF in milliseconds.
                Defaults to `5000`.
        """
        assert num_itpl > 2, (
            '`num_interpolate` (including the left and right classes) must be' +
            ' larger than 2.'
        )
        assert itpl_method in ['linspace', 'slerp'], (
            "`itpl_method` must be 'linspace' or 'slerp'"
        )

        super(MakeInterpolateSyntheticGIFCallback, self).__init__(
            filename=filename,
            nrows=None,
            ncols=None,
            postprocess_fn=postprocess_fn,
            normalize=normalize,
            latent_dim=latent_dim,
            image_dim=image_dim,
            keep_noise=keep_noise,
            seed=seed,
            delete_png=delete_png,
            duration=duration,
            **kwargs
        )
        self.itpl_method = itpl_method
        self.start_classes = start_classes
        self.stop_classes = stop_classes
        self.num_itpl = num_itpl
        self.num_classes = num_classes
        self.class_names = class_names
        # Reset unused inherited attributes
        self.save_freq = None

    def on_epoch_end(self, epoch, logs=None):
        # Deactivate MakeSyntheticGIFCallback.on_epoch_end()
        pass
    
    def on_train_end(self, logs=None):
        # Interpolate from start- to stop-classes
        itpl_ratios = torch.linspace(start=0, end=1, steps=self.num_itpl, dtype=torch.float).numpy().tolist()
        for ratio in itpl_ratios:
            label = self._interpolate(start=self.start, stop=self.stop, ratio=ratio)
            self.label = torch.cat(torch.split(label), axis=0)
            x_synth = self.synthesize_images()
            self.make_figure(x_synth, ratio)

        # Make GIF
        path_png = f'{self.path_png_folder}/*.png'
        img, *imgs = [PIL.Image.open(f) for f in sorted(glob.glob(path_png))]
        img.save(
            fp=self.filename,
            format='GIF',
            append_images=imgs,
            save_all=True,
            duration=self.duration/(len(imgs) + 1),
            loop=0)

        if self.delete_png is True:
            for png in glob.glob(path_png):
                os.remove(png)
            os.rmdir(self.path_png_folder)

    def handle_args(self):
        super().handle_args()

        if self.host.onehot_label is None:
            warnings.warn(
                f'Host does not have attribute `onehot_label`. ' +
                'Proceed with assumption that it receives one-hot encoded inputs.')
            self.onehot_label = True
        elif self.host.onehot_label is not None:
            assert self.host.onehot_label is True, (
                'Callback only works with models receiving one-hot encoded inputs.'
            )
            self.onehot_label = True

        if self.num_classes is None:
            self.num_classes:int = self.host.num_classes

        if self.class_names is None:
            self.class_names = [f'Class {i}' for i in range(self.num_classes)]

        # Parse interpolate method, start_classes and stop_classes
        if self.itpl_method == 'linspace':
            self._interpolate = self.linspace
        elif self.itpl_method == 'slerp':
            self._interpolate = self.slerp

        if self.start_classes is None:
            self.start_classes = [label for label in range(self.num_classes)]
        if self.stop_classes is None:
            self.stop_classes = [label for label in range(self.num_classes)]
        self.start_classes = torch.tensor(self.start_classes, dtype=torch.long)
        self.stop_classes = torch.tensor(self.stop_classes, dtype=torch.long)

        self.nrows = self.start_classes.shape[0]
        self.ncols = self.stop_classes.shape[0]

    def precompute_inputs(self):
        super(MakeInterpolateSyntheticGIFCallback, self).precompute_inputs()
        # Convert to one-hot labels
        start = torch.nn.functional.one_hot(input=self.start_classes, num_classes=self.num_classes).to(dtype=torch.float)
        stop = torch.nn.functional.one_hot(input=self.stop_classes, num_classes=self.num_classes).to(dtype=torch.float)

        # Expand dimensions to have shape [nrows, ncols, num_classes]
        start = torch.unsqueeze(input=start, dim=1)
        start = torch.repeat_interleave(input=start, repeats=self.ncols, dim=1)
        stop = torch.unsqueeze(input=stop, dim=0)
        stop = torch.repeat_interleave(input=stop, repeats=self.nrows, dim=0)

        self.start = start
        self.stop = stop

        if self.itpl_method == 'slerp':
            # Normalize (L2) to [-1, 1] for numerical stability
            norm_start = start/torch.linalg.norm(start, axis=-1)
            norm_stop = stop/torch.linalg.norm(stop, axis=-1)

            dotted = (norm_start*norm_stop).sum(axis=-1)
            # Clip to [-1, 1] for numerical stability
            clipped = torch.clamp(dotted, -1, 1)
            omegas = torch.acos(clipped)
            sinned = torch.sin(omegas)

            # Expand dimensions to have shape [nrows, ncols, num_classes]
            omegas = torch.unsqueeze(input=omegas, dim=-1)
            omegas = torch.repeat_interleave(input=omegas, repeats=self.num_classes, dim=-1)
            sinned = torch.unsqueeze(input=sinned, dim=-1)
            sinned = torch.repeat_interleave(input=sinned, repeats=self.num_classes, dim=-1)
            zeros_mask = (omegas == 0)

            self.omegas = omegas
            self.sinned = sinned
            self.zeros_mask = zeros_mask

    def synthesize_images(self):
        if self.keep_noise is False:
            batch_size = self.nrows*self.ncols
            self.latent_noise = torch.normal(mean=0, std=1, size=[batch_size, self.latent_dim])
        
        self.host.generator.eval()
        with torch.inference_mode():
            x_synth = self.host.generator(self.latent_noise, self.label)
            x_synth = self.postprocess_fn(x_synth)
        return x_synth

    def modify_suptitle(self, figure:Figure, value:float):
        figure.suptitle(f'{self.host.__class__.__name__} - {self.itpl_method} interpolation: {value*100:.2f}%')

    def modify_axis(self, axis:Axes):
        xticks = (self.image_dim[1] + 1)*np.arange(len(self.stop_classes)) + self.image_dim[1]/2
        xticklabels = [self.class_names[label] for label in self.stop_classes]

        yticks = (self.image_dim[0] + 1)*np.arange(len(self.start_classes)) + self.image_dim[0]/2
        yticklabels = [self.class_names[label] for label in self.start_classes]
        
        axis.set_frame_on(False)
        axis.tick_params(axis='both', length=0)
        axis.set(
            xlabel='Stop classes', xticks=xticks, xticklabels=xticklabels,
            ylabel='Start classes', yticks=yticks, yticklabels=yticklabels)

    def modify_savepath(self, value:float):
        return f"{self.path_png_folder}/{self.host.__class__.__name__}_itpl_{value:.4f}.png"

    def linspace(self, start, stop, ratio:float):
        label = ((1-ratio)*start + ratio*stop)
        return label
    
    def slerp(self, start, stop, ratio:float):
        label = torch.where(
            condition=self.zeros_mask,
            # Normal case: omega(s) != 0
            x=self.linspace(start=start, stop=stop, ratio=ratio),
            # Special case: omega(s) == 0 --> Use L'Hospital's rule for sin(0)/0
            y=(  torch.sin((1-ratio)*self.omegas) / self.sinned * start
               + torch.sin(ratio    *self.omegas) / self.sinned * stop
            )
        )
        return label
