import torch

class Kernel(torch.nn.Module):
    """Base class for kernels."""
    def __init__(self):
        super().__init__()

class RadialBasisFunctionKernel(Kernel):
    """Radial Basis Function kernel (a.k.a Squared Exponential or Gaussian).
    
    Args:
        `variance`: variance. Defaults to `1`.
        `length_scale`: length scale. Defaults to `1`.
    """
    def __init__(self, variance:float=1, length_scale:float=1):
        super().__init__()
        self.variance = variance
        self.length_scale = length_scale

    def __repr__(self):
        return f'{self.__class__.__name__}(variance={self.variance}, length_scale={self.length_scale})'

    def forward(self, input_1:torch.Tensor, input_2:torch.Tensor) -> torch.Tensor:
        broadcast_1 = input_1.unsqueeze(dim=1)
        broadcast_2 = input_2.unsqueeze(dim=0)

        x = self.variance*torch.exp(
            -1/(2*self.length_scale**2)
            * torch.sum((broadcast_1 - broadcast_2)**2, dim=-1)
        )
        return x

class ConstantKernel(Kernel):
    def __init__(self, constant:float=0):
        super().__init__()
        self.constant = constant
    
    def __repr__(self):
        return f'{self.__class__.__name__}(constant={self.constant})'

    def forward(self, input_1:torch.Tensor, input_2:torch.Tensor) -> torch.Tensor:
        return self.constant*torch.ones(size=(input_1.shape[0], input_2.shape[0]))

class LinearKernel(Kernel):
    def __repr__(self):
        return f'{self.__class__.__name__}()'
    
    def forward(self, input_1:torch.Tensor, input_2:torch.Tensor) -> torch.Tensor:
        broadcast_1 = input_1.unsqueeze(dim=1)
        broadcast_2 = input_2.unsqueeze(dim=0)
        x = torch.sum((broadcast_1 * broadcast_2), dim=-1)
        return x

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def plot_kernel(kernel):
        BOUNDS = [-10, 10]
        x = torch.linspace(start=BOUNDS[0], end=BOUNDS[1], steps=41).unsqueeze(dim=1)
        zeros = torch.zeros(size=(1, *x.shape[1:]))
        ones = torch.ones(size=(1, *x.shape[1:]))
        
        fig, ax = plt.subplots(nrows=1, ncols=3, constrained_layout=True, squeeze=False)
        ax[0, 0].plot(kernel(x, zeros))
        ax[0, 0].set(
            title=f'{kernel}\nbetween x and 0s',
            xticks=torch.linspace(start=0, end=x.shape[0]-1, steps=6),
            xticklabels=torch.linspace(start=BOUNDS[0], end=BOUNDS[1], steps=6).numpy(),
        )
        ax[0, 1].plot(kernel(x, ones))
        ax[0, 1].set(
            title=f'{kernel}\nbetween x and 1s',
            xticks=torch.linspace(start=0, end=x.shape[0]-1, steps=6),
            xticklabels=torch.linspace(start=BOUNDS[0], end=BOUNDS[1], steps=6).numpy(),
        )
        ax[0, 2].imshow(kernel(x, x))
        ax[0, 2].set(
            title=f'{kernel}\nbetween x and x',
            xticks=torch.linspace(start=0, end=x.shape[0]-1, steps=6),
            xticklabels=torch.linspace(start=BOUNDS[0], end=BOUNDS[1], steps=6).numpy(),
            yticks=torch.linspace(start=0, end=x.shape[0]-1, steps=6),
            yticklabels=torch.linspace(start=BOUNDS[0], end=BOUNDS[1], steps=6).numpy(),
        )
        return fig, ax

    rbf_k = RadialBasisFunctionKernel(variance=3, length_scale=2)
    plot_kernel(rbf_k)

    const_k = ConstantKernel(constant=17)
    plot_kernel(const_k)

    lin_k = LinearKernel()
    plot_kernel(lin_k)

    plt.show()