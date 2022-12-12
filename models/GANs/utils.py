from typing import Union, List, Callable, Any
import glob
from typing import Callable, Union, List, Any
import os
import PIL
import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if __name__ == '__main__':
    # Change path
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
    sys.path.append(repo_path)

from callbacks.Callbacks import Callback

class Reshape(torch.nn.Module):
    def __init__(self, out_shape:List[int]):
        super().__init__()
        self.out_shape = out_shape
    
    def forward(self, x:torch.Tensor):
        return x.view([-1, *self.out_shape])

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
                 image_dim:Union[None, List[int]]=None,
                 keep_noise:bool=True,
                 seed:Union[None, int]=None,
                 delete_png:bool=True,
                 save_freq:int=1,
                 duration:float=5000,
                 **kwargs):
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
        super().__init__(**kwargs)
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
            self.make_figure(x_synth.clone().detach(), epoch)

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
            self.latent_noise = torch.normal(mean=0, std=1, size=(batch_size, self.latent_dim))
        elif self.seed is not None:
            with torch.manual_seed(self.seed):
                self.latent_noise = torch.normal(mean=0, std=1, size=(batch_size, self.latent_dim))

    def synthesize_images(self) -> torch.Tensor:
        """Produce synthetic images with the generator.
        
        Returns:
            A batch of synthetic images.
        """
        if self.keep_noise is False:
            batch_size = self.nrows*self.ncols
            if self.seed is None:
                self.latent_noise = torch.normal(mean=0, std=1, size=(batch_size, self.latent_dim))
            elif self.seed is not None:
                with torch.manual_seed(self.seed):
                    self.latent_noise = torch.normal(mean=0, std=1, size=(batch_size, self.latent_dim))
        
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