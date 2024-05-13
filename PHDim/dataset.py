import torch
from torchvision import datasets, transforms

from PHDim.eval import recover_eval_tensors


class QuickDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, device='cuda' if torch.cuda.is_available() else 'cpu', random=False):
        super().__init__()
        self.x = x.to(device)
        self.y = y.to(device)

        if random:
            self.y = self.y[torch.randperm(len(self.y))]
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class DataOptions:

    def __init__(self, dataset, path, bs_train, bs_eval, resize=None):
        self.dataset = dataset
        self.path = path
        self.batch_size_train = bs_train
        self.batch_size_eval = bs_eval
        self.resize = resize


def get_data_simple(dataset, path, bs_train, bs_eval, subset=None, resize=None):

    return get_data(DataOptions(
        dataset,
        path,
        bs_train,
        bs_eval,
        resize
    ), subset_percentage=subset)


def get_data(args: DataOptions, subset_percentage: float = None):

    # mean/std stats
    if args.dataset == 'cifar10':
        data_class = 'CIFAR10'
        num_classes = 10
        stats = {
            'mean': [0.491, 0.482, 0.447],
            'std': [0.247, 0.243, 0.262]
        }
    elif args.dataset == 'cifar100':
        data_class = 'CIFAR100'
        num_classes = 100
        stats = {
            'mean': [0.5071, 0.4867, 0.4408],
            'std': [0.2675, 0.2565, 0.2761]
        }
    elif args.dataset == 'mnist':
        data_class = 'MNIST'
        num_classes = 10
        stats = {
            'mean': [0.1307],
            'std': [0.3081]
        }
    else:
        raise ValueError("unknown dataset")

    # input transformation w/o preprocessing for now

    trans = [
        transforms.ToTensor(),
        lambda t: t.type(torch.get_default_dtype()),
        transforms.Normalize(**stats)
    ]

    # get train and test data with the same normalization
    tr_data = getattr(datasets, data_class)(
        root=args.path,
        train=True,
        download=True,
        transform=transforms.Compose(trans)
    )

    te_data = getattr(datasets, data_class)(
        root=args.path,
        train=False,
        download=True,
        transform=transforms.Compose(trans)
    )

    # get tr_loader for train/eval and te_loader for eval
    temp_train_loader = torch.utils.data.DataLoader(
        dataset=tr_data,
        batch_size=args.batch_size_train,
    )

    temp_test_loader = torch.utils.data.DataLoader(
        dataset=te_data,
        batch_size=args.batch_size_eval,
    )

    train_x, train_y = recover_eval_tensors(temp_train_loader)
    test_x, test_y = recover_eval_tensors(temp_test_loader)

    train_dataset = QuickDataset(train_x, train_y)
    test_dataset = QuickDataset(test_x, test_y)

    train_dataset_random = QuickDataset(train_x, train_y, random=True)

    # get tr_loader for train/eval and te_loader for eval
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size_train,
        shuffle=True,
    )

    # get tr_loader for train/eval and te_loader for eval
    train_loader_eval = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size_eval,
        shuffle=False,
    )

    test_loader_eval = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size_eval,
        shuffle=False,
    )

    # get tr_loader for train/eval and te_loader for eval
    train_loader_random = torch.utils.data.DataLoader(
        dataset=train_dataset_random,
        batch_size=args.batch_size_train,
        shuffle=True,
    )

    # get tr_loader for train/eval and te_loader for eval
    train_loader_random_eval = torch.utils.data.DataLoader(
        dataset=train_dataset_random,
        batch_size=args.batch_size_eval,
        shuffle=False,
    )




    return train_loader, test_loader_eval, train_loader_eval, num_classes, train_loader_random, train_loader_random_eval
