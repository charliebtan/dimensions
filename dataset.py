import torch
from torchvision import datasets, transforms
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset

DATA_STATS = {
    'MNIST': {
        'mean': [0.1307],
        'std': [0.3081]
    },
    'CIFAR10': {
        'mean': [0.491, 0.482, 0.447],
        'std': [0.247, 0.243, 0.262]
    },
    'CIFAR100': {
        'mean': [0.507, 0.487, 0.441],
        'std': [0.268, 0.257, 0.276]
    }
}

class QuickDataset(Dataset):
    """Is faster for small datasets than the default PyTorch Dataset class."""
    def __init__(self, x: torch.Tensor, y: torch.Tensor, device: str = 'cuda' if torch.cuda.is_available() else 'cpu', random: bool = False):
        super().__init__()
        self.x = x.to(device)
        self.y = y.to(device)

        if random:
            self.y = self.y[torch.randperm(len(self.y))]
    
    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]

def get_data_as_tensors(dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Passes through the dataloader, applying any transformations and returns the data as tensors."""

    all_x, all_y = [], []

    for x, y in dataloader:
        all_x.append(x)
        all_y.append(y)

    return torch.cat(all_x), torch.cat(all_y)

def prepare_data(dataset: str, path: str, batch_size: int, batch_size_eval: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare the data for training and evaluation."""

    dataset = dataset.upper()

    if dataset not in ["MNIST", "CIFAR10", "CIFAR100"]:
        raise NotImplementedError(f"Dataset {dataset} not implemented, should be in ['mnist', 'cifar10', 'cifar100']")

    # Input transformations
    transform = [
        transforms.ToTensor(),
        lambda t: t.type(torch.get_default_dtype()),
        transforms.Normalize(**DATA_STATS[dataset])
    ]

    # Load the data
    train_data: VisionDataset = getattr(datasets, dataset.upper())(
        root=path,
        train=True,
        download=True,
        transform=transforms.Compose(transform)
    )
    test_data: VisionDataset = getattr(datasets, dataset.upper())(
        root=path,
        train=False,
        download=True,
        transform=transforms.Compose(transform)
    )

    # Get the data as tensors for QuickDataset
    temp_train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
    )
    temp_test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
    )
    train_x, train_y = get_data_as_tensors(temp_train_loader)
    test_x, test_y = get_data_as_tensors(temp_test_loader)

    # QuickDataset for faster data loading of small datasets
    train_dataset = QuickDataset(train_x, train_y)
    test_dataset = QuickDataset(test_x, test_y)

    # Train loader for training
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Train loader for evaluation (larger batch size)
    train_loader_eval = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
    )

    # Test loader for evaluation (larger batch size)
    test_loader_eval = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
    )

    return train_loader, train_loader_eval, test_loader_eval

def cycle_loader(dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Infinite loop over the dataloader."""
    while True:
        for data in dataloader:
            yield data
