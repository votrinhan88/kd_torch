import os, sys
repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
if sys.path[0] != repo_path:
    sys.path.insert(0, repo_path)

from typing import Sequence
import torch

from models.bayes_opt.gaussian_process import GaussianProcess

class AcquisitionFunction(torch.nn.Module):
    """Base class for acquisition functions."""
    def __init__(self, gp:GaussianProcess):
        super().__init__()
        self.gp = gp

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def forward(self) -> torch.Tensor:
        pass

class UpperConfidenceBound(AcquisitionFunction):
    def __init__(self, gp:GaussianProcess, beta:float=0.1):
        super().__init__(gp=gp)
        self.beta = beta

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(beta={self.beta})'

    def forward(self) -> torch.Tensor:
        mean = self.gp.mean_posterior.squeeze(dim=1)
        std = self.gp.covar_posterior.diag().sqrt()
        return mean + self.beta*std

class ExpectedImprovement(AcquisitionFunction):
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def forward(self) -> torch.Tensor:
        mean = self.gp.mean_posterior.squeeze(dim=1)
        std = self.gp.covar_posterior.diag().sqrt()
        best = mean.max()

        z = (mean - best) / std
        distr = torch.distributions.normal.Normal(loc=0, scale=1)
        ei = (mean - best)*distr.cdf(z) + std*distr.log_prob(z).exp()
        return ei

class ProbabilityOfImprovement(AcquisitionFunction):
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def forward(self) -> torch.Tensor:
        mean = self.gp.mean_posterior.squeeze(dim=1)
        std = self.gp.covar_posterior.diag().sqrt()
        best = mean.max()

        z = (mean - best) / std
        distr = torch.distributions.normal.Normal(loc=0, scale=1)
        poi = distr.cdf(z)
        return poi

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from models.bayes_opt.gaussian_process import FixedNoiseGaussianProcess
    from models.bayes_opt.kernels import RadialBasisFunctionKernel

    def expt_acquisition_fn():
        # Hparams for dummy data
        NUM_OBSERVED = 5                # Number of observable datapoints
        SIGMA_NOISE_OBSERVED = 0.05     # Standard deviation of Gaussian noise added to dummy data
        # Hparams for building Prior
        BOUNDS = [[0, 10]]
        RANGE_EXTEND = 0.2              # How far out of the observed domain the Posterior should extend to
        NUM_SAMPLE = 51                 # Number of sampling points to visualize the Prior and Posterior
        LENGTH = 1                      # Parameter for Kernel
        VARIANCE = 0.1                  # Parameter for Kernel
        SIGMA_NOISE_POSTERIOR = 1e-3    # Parameter for Posterior

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
            x = x.unsqueeze(dim=1)
            return x
        x_sample = make_sample_for_prior(BOUNDS, RANGE_EXTEND, NUM_SAMPLE)

        # Fit a Gaussian process
        gp = FixedNoiseGaussianProcess(
            x_train=x_train,
            y_train=y_train,
            kernel=RadialBasisFunctionKernel(variance=VARIANCE, length_scale=LENGTH),
            noise=SIGMA_NOISE_POSTERIOR
        )
        gp.forward(x_sample)

        ucb = UpperConfidenceBound(gp=gp, beta=2)
        ei = ExpectedImprovement(gp=gp)
        poi = ProbabilityOfImprovement(gp=gp)

        # Visualize Prior and Posterior
        def plot_helper(
            x, mean, std,
            acq_fns:Sequence[AcquisitionFunction],
            x_train=None, y_train=None,
        ):
            num_acq = len(acq_fns)

            fig, ax = plt.subplots(num_acq+1, 1, sharex=True, constrained_layout=True, squeeze=False)
            ax[0, 0].autoscale()
            ax[0, 0].plot(x, mean, color='blue', label='Posterior')
            ax[0, 0].fill_between(x, mean - 2*std, mean + 2*std, color='red', alpha=0.5, label="$\pm$ 2 STD")
            # Acquisition functions
            for i, a in enumerate(acq_fns):
                ax[i+1, 0].plot(x, a(), label=a)
                ax[i+1, 0].set(title=f'{a}')

            ax[0, 0].scatter(x_train, y_train, color='black')
            ax[0, 0].legend()
            ax[0, 0].set(title='Posterior $\pm 2$ STD')
            return fig, ax

        ## Posterior
        mean = gp.mean_posterior
        std = gp.covar_posterior.diag().sqrt()
        plot_helper(
            x=x_sample.squeeze(), mean=mean.squeeze(), std=std.squeeze(), acq_fns=[ucb, ei, poi],
            x_train=gp.x_train.squeeze(), y_train=gp.y_train.squeeze()
        )

        plt.show()

    expt_acquisition_fn()