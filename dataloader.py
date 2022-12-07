from typing import Union, Callable, Any, Tuple, Literal, Optional

import torch
import torchvision
from torch.utils.data import DataLoader

DATASET_CLASS = {
    'MNIST':torchvision.datasets.MNIST,
    'FashionMNIST':torchvision.datasets.FashionMNIST,
    'CIFAR10':torchvision.datasets.CIFAR10,
    'CIFAR100':torchvision.datasets.CIFAR100,
}

def compute_mean_std(dataset:str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the channel-wise mean and standard deviation of a dataset, typically
    for rescaling.
    
    Args:
        `dataset`: Name of dataset.
    Returns:
        A tuple of `(mean, std)`.
    """
    trainset = DATASET_CLASS[dataset](
        root='./datasets',
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    num_examples = trainset.data.shape[0]
    trainloader = DataLoader(
        dataset=trainset,
        batch_size=num_examples,
        shuffle=False,
        drop_last=False,
    )

    x:torch.Tensor = next(iter(trainloader))[0]
    reduce_dim = [i for i in range(x.dim()) if i != 1]
    mean = x.mean(dim=reduce_dim)
    std = x.std(dim=reduce_dim)
    return mean, std

def get_dataloader(dataset:str,
                   augmentation_trans:Optional[Callable]=None,
                   resize:Optional[Tuple[float, float]]=None,
                   rescale:Optional[Literal['standardization']| Tuple[float, float]]=None,
                   batch_size_train:int=128,
                   batch_size_test:int=1024,
                   drop_last:bool=False,
                   num_workers:Optional[int]=0,
                   onehot_label:bool=False,
                   ):
    ROOT = './datasets'
    
    STANDARDIZATION_MEAN_STD = {
        'CIFAR10': (torch.Tensor([[[0.4914, 0.4822, 0.4465]]]), torch.Tensor([[[0.2470, 0.2435, 0.2616]]])),
        'CIFAR100': (torch.Tensor([[[0.5071, 0.4866, 0.4409]]]), torch.Tensor([[[0.2673, 0.2564, 0.2762]]])),
        'FashionMNIST': (torch.Tensor([0.2860]), torch.Tensor([0.3530])),
        'MNIST': (torch.Tensor([0.1307]), torch.Tensor([0.3081])),
    }
    def WORKER_INIT_FN(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset  # the dataset copy in this worker process
        overall_start = dataset.start
        overall_end = dataset.end
        # configure the dataset to only process the split workload
        per_worker = int(torch.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
        worker_id = worker_info.id
        dataset.start = overall_start + worker_id * per_worker
        dataset.end = min(dataset.start + per_worker, overall_end)

    transforms_train = []
    transforms_test = []

    # Augmentation
    if augmentation_trans is not None:
        transforms_train.append(augmentation_trans)
    # Resizing
    if resize is not None:
        resize = torchvision.transforms.Resize(size=resize)
        transforms_train.append(resize)
        transforms_test.append(resize)
    # Converting to tensors
    toTensor = torchvision.transforms.ToTensor()
    transforms_train.append(toTensor)
    transforms_test.append(toTensor)
    # Rescaling (Normalization)
    if rescale is not None:
        if rescale == 'standardization':
            mean, std = STANDARDIZATION_MEAN_STD[dataset]
        else:
            mean = -rescale[0]/(rescale[1] - rescale[0])
            std = 1/(rescale[1] - rescale[0])
        rescale = torchvision.transforms.Normalize(mean=mean, std=std)
        transforms_train.append(rescale)
        transforms_test.append(rescale)

    transforms_train = torchvision.transforms.Compose(transforms_train)
    transforms_test = torchvision.transforms.Compose(transforms_test)

    # One-hot label
    if onehot_label is True:
        NUM_CLASSES = len(DATASET_CLASS[dataset].classes)
        def target_transform(y):
            return torch.zeros(NUM_CLASSES).scatter_(0, torch.tensor(y), value=1)
    else:
        target_transform = None

    _dataset = {
        'train': DATASET_CLASS[dataset](
            root=ROOT,
            train=True,
            transform=transforms_train,
            target_transform=target_transform,
            download=True
        ),
        'test': DATASET_CLASS[dataset](
            root=ROOT,
            train=False,
            transform=transforms_test,
            target_transform=target_transform,
            download=True
        ),
    }

    dataloader = {
        'train': DataLoader(
            dataset=_dataset['train'],
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=num_workers,
            drop_last=drop_last,
            worker_init_fn=WORKER_INIT_FN
        ),
        'test': DataLoader(
            dataset=_dataset['test'],
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            worker_init_fn=None
        ),
    }
    return dataloader

if __name__ == '__main__':
    ds = get_dataloader(
        dataset='MNIST',
        rescale='standardization',
        onehot_label=True
    )
    print()