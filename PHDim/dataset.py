import torch
from torchvision import datasets, transforms

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

class QuickDataset(torch.utils.data.Dataset):
    """Is faster for small datasets than the default PyTorch Dataset class."""
    def __init__(self, x, y, device='cuda' if torch.cuda.is_available() else 'cpu', random=False): # TODO device
        super().__init__()
        self.x = x.to(device)
        self.y = y.to(device)

        if random:
            self.y = self.y[torch.randperm(len(self.y))]
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

def get_data_as_tensors(dataloader):
    """Passes through the dataloader, applying any transformations and returns the data as tensors."""

    all_x, all_y = [], []

    for x, y in dataloader:

        all_x.append(x)
        all_y.append(y)

    return torch.cat(all_x), torch.cat(all_y)


def prepare_data(dataset, path, batch_size, batch_size_eval):

    dataset = dataset.upper()

    if dataset not in ["MNIST", "CIFAR10", "CIFAR100"]:
        raise NotImplementedError(f"Dataset {dataset} not implemented, should be in ['mnist', 'cifar10', 'cifar100']")

    # Input transformations
    tranform = [
        transforms.ToTensor(),
        lambda t: t.type(torch.get_default_dtype()),
        transforms.Normalize(**DATA_STATS[dataset])
    ]

    # Load the data
    train_data = getattr(datasets, dataset.upper())(
        root=path,
        train=True,
        download=True,
        transform=transforms.Compose(tranform)
    )
    test_data = getattr(datasets, dataset.upper())(
        root=path,
        train=False,
        download=True,
        transform=transforms.Compose(tranform)
    )

    # Get the data as tensors for QuickDataset
    temp_train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
    )
    temp_test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
    )
    train_x, train_y = get_data_as_tensors(temp_train_loader)
    test_x, test_y = get_data_as_tensors(temp_test_loader)

    # QuickDataset for faster data loading of small datasets
    train_dataset = QuickDataset(train_x, train_y)
    test_dataset = QuickDataset(test_x, test_y)

    # Train loader for training
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Train loader for evaluation
    train_loader_eval = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
    )

    # Test loader for evaluation
    test_loader_eval = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
    )

    return train_loader, train_loader_eval, test_loader_eval

def cycle_loader(dataloader):
    """Infinite loop over the dataloader."""
    while 1:
        for data in dataloader:
            yield data
