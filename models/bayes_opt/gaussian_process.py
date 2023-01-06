from typing import Callable, Literal

import torch
import matplotlib.pyplot as plt
import numpy as np

class GaussianProcess(torch.nn.Module):
    """Base class for Gaussian process."""
    # For numerical stability
    EPSILON = 1e-7

    def __init__(self, x_train:torch.Tensor, y_train:torch.Tensor):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.mean_prior:torch.Tensor = None
        self.covar_prior:torch.Tensor = None
        self.mean_posterior:torch.Tensor = None
        self.covar_posterior:torch.Tensor = None

class FixedNoiseGaussianProcess(GaussianProcess):
    """Gaussian process.
    
    Args:
        `x_train`: Observed inputs of shape B x D.
        `y_train`: Observed labels of shape B x 1.
        `kernel`: The kernel.
        `noise`: Noise. Defaults to `0`.
    
    B: batch size
    D: dimension of sample (number of features)
    """
    def __init__(self,
        x_train:torch.Tensor,
        y_train:torch.Tensor,
        kernel:Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        noise:float=0,
    ):
        super().__init__(
            x_train=x_train,
            y_train=y_train,            
        )
        self.kernel = kernel
        self.noise = noise
        self.fit()
        
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    x_train:{tuple(self.x_train.shape)},\n"
            f"    y_train:{tuple(self.y_train.shape)},\n"
            f"    kernel={self.kernel},\n"
            f"    noise={self.noise}\n"
            f")"
        )
    
    def fit(self):
        # Normalize to mean = 0
        self.mean_y_train = self.y_train.mean()
        self.norm_y_train = self.y_train - self.mean_y_train

        num_observed = self.x_train.shape[0]
        self.covar_observed = (
            self.kernel(self.x_train, self.x_train)
            + (self.noise + self.EPSILON)*torch.eye(num_observed)
        )
        
    def forward(self, x:torch.Tensor):
        # Assuming a mean of 0 for simplicity
        self.mean_prior = torch.zeros(size=[x.shape[0], 1])
        self.covar_prior = self.kernel(x, x)

        covar_11_inv = self.covar_observed.inverse()
        covar_22 = self.covar_prior
        covar_21 = self.kernel(x, self.x_train) # == covar_12.T

        ## Mean (added back the input's mean) & Variance of posterior
        self.mean_posterior = (covar_21 @ covar_11_inv @ self.norm_y_train) + self.mean_y_train
        self.covar_posterior = covar_22 - covar_21 @ covar_11_inv @ covar_21.T
        return self.mean_posterior, self.covar_posterior

    def realize_prior(self, num_realize:int=1):
        """Draw samples from the prior.
        
        Args:
            `num_realize`: Number of realizations. Defaults to `1`.
        """
        y = np.random.multivariate_normal(
            mean=self.mean_prior.squeeze(dim=1),
            cov=self.covar_prior,
            size=num_realize
        )
        return y

    def realize_posterior(self, num_realize:int=1):
        """Draw samples from the posterior.
        
        Args:
            `num_realize`: Number of realizations. Defaults to `1`.
        """
        y = np.random.multivariate_normal(
            mean=self.mean_posterior.squeeze(dim=1),
            cov=self.covar_posterior,
            size=num_realize
        )
        return y

    def update(self, x_train:torch.Tensor, y_train:torch.Tensor):
        """Update the Gaussian process with new data. 
        
        Args:
            `x_train`: Additional observed inputs of shape B x D.
            `y_train`: Additional observed labels of shape B x 1.
        
        B: batch size
        D: dimension of sample (number of features)
        """        
        self.x_train = torch.cat([self.x_train, x_train], dim=0)
        self.y_train = torch.cat([self.y_train, y_train], dim=0)
        self.fit()
        
if __name__ == '__main__':
    # Change path
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
    if sys.path[0] != repo_path:
        sys.path.insert(0, repo_path)

    from models.bayes_opt.kernels import RadialBasisFunctionKernel

    def expt_gaussian_process_1d():
        # Hparams for dummy data
        NUM_OBSERVED = 5                # Number of observable datapoints
        EXTRA_OBSERVED = 3              # Number of extra observable datapoints
        SIGMA_NOISE_OBSERVED = 0.05     # Standard deviation of Gaussian noise added to dummy data
        # Hparams for building Prior
        BOUNDS = [[0, 10]]
        RANGE_EXTEND = 0.2              # How far out of the observed domain the Posterior should extend to
        NUM_SAMPLE = 41                 # Number of sampling points to visualize the Prior and Posterior
        LENGTH = 1                      # Parameter for Kernel
        VARIANCE = 0.1                  # Parameter for Kernel
        SIGMA_NOISE_POSTERIOR = 1e-3    # Parameter for Posterior
        # Hparams for realization
        NUM_REALIZE = 5                 # Number of functions to be realized

        # Make train data
        BOUNDS = torch.tensor(BOUNDS)
        MEAN = -BOUNDS[:, 0]/(BOUNDS[:, 1] - BOUNDS[:, 0])
        STD = 1/(BOUNDS[:, 1] - BOUNDS[:, 0])
        x_train = (torch.rand(size = [NUM_OBSERVED, 1]) - MEAN)/STD
        def make_label(x:torch.Tensor) -> torch.Tensor:
            return torch.sin(x) + 2 + torch.normal(mean=0, std=SIGMA_NOISE_OBSERVED, size=x.size())
        y_train = make_label(x_train)

        # Make dummy sampling points for prior with extend bounds 
        def make_sample_for_prior(bounds:torch.Tensor, extend:float, num_sample:int):
            p2p = bounds[:, 1] - bounds[:, 0]
            x:torch.Tensor = torch.linspace(
                start=(bounds[:, 0] - extend*p2p)[0],
                end=(bounds[:, 1] + extend*p2p)[0],
                steps=num_sample
            )
            x = x.unsqueeze(dim = 1)
            return x
        x_sample = make_sample_for_prior(BOUNDS, RANGE_EXTEND, NUM_SAMPLE)
        
        # Fit a Gaussian process and render
        gp = FixedNoiseGaussianProcess(
            x_train=x_train,
            y_train=y_train,
            kernel=RadialBasisFunctionKernel(variance=VARIANCE, length_scale=LENGTH),
            noise=SIGMA_NOISE_POSTERIOR
        )
        gp.forward(x_sample)

        # Visualize Prior and Posterior
        def plot_helper(
            x, ys, mean, std,
            option:Literal['prior', 'posterior'], x_train=None, y_train=None,
        ):
            label = option[0].capitalize() + option[1:]

            fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
            ax[0].autoscale()
            ax[0].plot(x, mean, color='blue', label=label)
            ax[0].fill_between(x, mean - 2*std, mean + 2*std, color='red', alpha=0.5, label="$\pm$ 2 STD")
            ax[0].legend()
            ax[0].set_title(f'{label} $\pm 2$ STD')
            # Some realizations
            for rlz in torch.arange(NUM_REALIZE):
                ax[1].plot(x, ys[rlz, :])
            ax[1].set_title(f'{NUM_REALIZE} realizations of the {label}')
            if option == 'posterior':
                ax[0].scatter(x_train, y_train, color='black')
                ax[1].scatter(x_train, y_train, color='black')

            return fig, ax
        
        ## Prior
        mean = gp.mean_prior
        std = gp.covar_prior.diag().sqrt()
        ys = gp.realize_prior(NUM_REALIZE)
        plot_helper(x_sample.squeeze(), ys, mean.squeeze(), std.squeeze(), 'prior')

        ## Posterior
        mean = gp.mean_posterior
        std = gp.covar_posterior.diag().sqrt()
        ys = gp.realize_posterior(NUM_REALIZE)
        plot_helper(x_sample.squeeze(), ys, mean.squeeze(), std.squeeze(), 'posterior', gp.x_train.squeeze(), gp.y_train.squeeze())

        ## More training data
        extra_x_train = (torch.rand(size = [EXTRA_OBSERVED, 1]) - MEAN)/STD
        extra_y_train = make_label(extra_x_train)
        gp.update(extra_x_train, extra_y_train)
        gp.forward(x_sample)

        mean = gp.mean_posterior
        std = gp.covar_posterior.diag().sqrt()
        ys = gp.realize_posterior(NUM_REALIZE)
        plot_helper(x_sample.squeeze(), ys, mean.squeeze(), std.squeeze(), 'posterior', gp.x_train.squeeze(), gp.y_train.squeeze())

        plt.show()

    def expt_gaussian_process_2d():
        # Hparams for dummy data
        NUM_OBSERVED = 50               # Number of observable datapoints
        EXTRA_OBSERVED = 30             # Number of extra observable datapoints
        SIGMA_NOISE_OBSERVED = 0.05     # Standard deviation of Gaussian noise added to dummy data
        # Hparams for building Prior
        BOUNDS = [[-5, 5], [-5, 5]]
        RANGE_EXTEND = 0.2              # How far out of the observed domain the Posterior should extend to
        NUM_SAMPLE = 21                 # Number of sampling points to visualize the Prior and Posterior (in each dimension)
        LENGTH = 1                      # Parameter for Kernel
        VARIANCE = 0.1                  # Parameter for Kernel
        SIGMA_NOISE_POSTERIOR = 1e-3    # Parameter for Posterior
        # Hparams for realization
        NUM_REALIZE = 1                 # Number of functions to be realized

        # Make train data
        BOUNDS = torch.tensor(BOUNDS)
        MEAN = -BOUNDS[:, 0]/(BOUNDS[:, 1] - BOUNDS[:, 0])
        STD = 1/(BOUNDS[:, 1] - BOUNDS[:, 0])
        x_train = (torch.rand(size = [NUM_OBSERVED, 2]) - MEAN)/STD
        def circular_wave(
            input:torch.Tensor,
            time:float=0.5,
            frequency:float=1,
            wavelength:float=10,
            amplitude:float=2,
            noise:float=0,
        ) -> torch.Tensor:
            pi = torch.acos(torch.zeros(1)).item()*2
            radius = (input[:, 0]**2 + input[:, 1]**2).sqrt()
            u = amplitude*torch.cos((2*pi/wavelength)*radius+(2*pi*frequency)*time)
            u = u + torch.normal(mean=0, std=noise, size=u.size())
            return u.unsqueeze(dim=1)
        y_train = circular_wave(x_train, noise=SIGMA_NOISE_OBSERVED)

        # Make dummy sampling points for prior with extend bounds 
        def make_sample_for_prior(bounds:torch.Tensor, extend:float, num_sample:int):
            p2p = bounds[:, 1] - bounds[:, 0]
            coords = torch.zeros(size=(bounds.shape[0], num_sample))
            for dim in range(bounds.shape[1]):
                coords[dim, :] = torch.linspace(
                    start=bounds[dim, 0] - extend*p2p[dim],
                    end=bounds[dim, 1] + extend*p2p[dim],
                    steps=num_sample
                )
            x = torch.cartesian_prod(*coords.unbind(dim=0))
            return x
        x_sample = make_sample_for_prior(BOUNDS, RANGE_EXTEND, NUM_SAMPLE)
        
        # Fit a Gaussian process and render
        gp = FixedNoiseGaussianProcess(
            x_train=x_train,
            y_train=y_train,
            kernel=RadialBasisFunctionKernel(variance=VARIANCE, length_scale=LENGTH),
            noise=SIGMA_NOISE_POSTERIOR
        )
        gp.forward(x_sample)

        # Visualize Prior and Posterior
        def plot_helper(
            x, ys, mean, std,
            option:Literal['prior', 'posterior'], x_train=None, y_train=None,
        ):
            x = x.reshape([NUM_SAMPLE, NUM_SAMPLE, x.shape[1]])
            ys = ys.reshape([ys.shape[0], NUM_SAMPLE, NUM_SAMPLE])
            mean = mean.reshape([NUM_SAMPLE, NUM_SAMPLE])
            std = std.reshape([NUM_SAMPLE, NUM_SAMPLE])
            label = option[0].capitalize() + option[1:]

            mean = mean.reshape([NUM_SAMPLE, NUM_SAMPLE])

            fig, ax = plt.subplots(
                nrows=1, ncols=2, subplot_kw={"projection": "3d"},
                constrained_layout=True, squeeze=False, sharex=True, sharey=True
            )

            ax[0, 0].plot_surface(x[:, :, 0], x[:, :, 1], mean, color='blue', alpha = 0.2, label=label)
            ax[0, 0].plot_surface(x[:, :, 0], x[:, :, 1], mean + 2*std, color='red', alpha = 0.2, label="+ 2 STD")
            ax[0, 0].plot_surface(x[:, :, 0], x[:, :, 1], mean - 2*std, color='red', alpha = 0.2, label="- 2 STD")
            ax[0, 0].set_title(f'{label} $\pm 2$ STD')

            # Some realizations
            for rlz in torch.arange(NUM_REALIZE):
                ax[0, 1].plot_surface(x[:, :, 0], x[:, :, 1], ys[rlz, :, :], alpha = 0.2)
            ax[0, 1].set_title(f'{NUM_REALIZE} realizations of the {label}')

            if option == 'posterior':
                ax[0, 0].scatter(x_train[:, 0], x_train[:, 1], y_train, color='black')
                ax[0, 1].scatter(x_train[:, 0], x_train[:, 1], y_train, color='black')
            return fig, ax
        
        ## Raw data
        x_true = x_sample.reshape([NUM_SAMPLE, NUM_SAMPLE, x_sample.shape[1]])
        y_true = circular_wave(x_sample, noise=SIGMA_NOISE_OBSERVED).reshape([NUM_SAMPLE, NUM_SAMPLE])
        fig, ax = plt.subplots(
            nrows=1, ncols=1, subplot_kw={"projection": "3d"},
            constrained_layout=True, squeeze=False
        )
        ax[0, 0].plot_surface(x_true[:, :, 0], x_true[:, :, 1], y_true, color='blue', alpha = 0.2)

        ## Prior
        mean = gp.mean_prior
        std = gp.covar_prior.diag().sqrt()
        ys = gp.realize_prior(NUM_REALIZE)
        plot_helper(x_sample, ys, mean, std, 'prior')

        ## Posterior
        mean = gp.mean_posterior
        std = gp.covar_posterior.diag().sqrt()
        ys = gp.realize_posterior(NUM_REALIZE)
        plot_helper(x_sample.squeeze(), ys, mean.squeeze(), std.squeeze(), 'posterior', gp.x_train.squeeze(), gp.y_train.squeeze())

        ## More training data
        extra_x_train = (torch.rand(size = [EXTRA_OBSERVED, 2]) - MEAN)/STD
        extra_y_train = circular_wave(extra_x_train, noise=SIGMA_NOISE_OBSERVED)
        gp.update(extra_x_train, extra_y_train)
        gp.forward(x_sample)

        mean = gp.mean_posterior
        std = gp.covar_posterior.diag().sqrt()
        ys = gp.realize_posterior(NUM_REALIZE)
        plot_helper(x_sample.squeeze(), ys, mean.squeeze(), std.squeeze(), 'posterior', gp.x_train.squeeze(), gp.y_train.squeeze())

        plt.show()

    expt_gaussian_process_1d()
    expt_gaussian_process_2d()