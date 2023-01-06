from typing import Callable, Literal, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset
import torchvision

def make_SmallMNIST(examples_per_class:int=5):
    # Due to setting seed manually, making the dataset and calling it for use
    # must be done in different runtimes
    NUM_CLASSES = 10
    NUM_EXAMPLES = 60000
    SEED = 17
    torch.manual_seed(SEED)

    mnist = torchvision.datasets.MNIST(
        root='./datasets',
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    mnistloader = DataLoader(
        dataset=mnist,
        batch_size=NUM_EXAMPLES,
        shuffle=False,
        drop_last=False,
    )

    x, y = next(iter(mnistloader))

    # LOCKED: Do not change! Will break reproducibility. #######################
    indices = {}
    for label in range(NUM_CLASSES):
        full_indices = (y==label).nonzero()
        full_indices = full_indices[torch.randperm(full_indices.shape[0])]
        indices[label] = full_indices[0:examples_per_class]
    indices = torch.cat([indices[label] for label in range(NUM_CLASSES)], dim=0).squeeze()
    # LOCKED (end) #############################################################

    torch.save(
        obj={
            'x': x,
            'y': y,
            'indices': indices
        },
        f='./datasets/SmallMNIST.pt'
    )

def get_SmallMNIST_loader(
    augmentation_trans:Optional[Callable]=None,
    resize:Optional[Tuple[float, float]]=None,
    rescale:Optional[Literal['standardization']| Tuple[float, float]]=None,
    batch_size_train:int=128,
    batch_size_val:int=1024,
    drop_last:bool=False,
    num_workers:Optional[int]=0,
    onehot_label:bool=False,
):
    ROOT = './datasets'
    NUM_CLASSES = 10
    STANDARDIZATION_MEAN_STD = (torch.Tensor([0.1307]), torch.Tensor([0.3081]))

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
            mean, std = STANDARDIZATION_MEAN_STD
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
        def target_transform(y):
            return torch.zeros(NUM_CLASSES).scatter_(0, torch.tensor(y), value=1)
    else:
        target_transform = None

    indices = torch.load('./datasets/SmallMNIST.pt')['indices']

    _dataset = {
        'train': Subset(
            dataset=torchvision.datasets.MNIST(
                root=ROOT,
                train=True,
                transform=transforms_train,
                target_transform=target_transform,
                download=True),
            indices=indices,
        ),
        'val': torchvision.datasets.MNIST(
            root=ROOT,
            train=False,
            transform=transforms_test,
            target_transform=target_transform,
            download=True,
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
        'val': DataLoader(
            dataset=_dataset['val'],
            batch_size=batch_size_val,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            worker_init_fn=None
        ),
    }
    return dataloader

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # make_SmallMNIST(5)
    dataloader = get_SmallMNIST_loader(batch_size_train=50, drop_last=False)
    x, y = next(iter(dataloader['train']))

    fig, ax = plt.subplots(nrows=5, ncols=10)
    for row in range(5):
        for col in range(10):
            id = row*10 + col
            ax[row, col].imshow(x[id, 0], cmap='gray')
    plt.show()