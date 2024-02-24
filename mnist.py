import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
import torch
import params
from transformations import *


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])

transform_replace_all = transforms.Compose([transforms.ToTensor(),
                                RandomSingleColorReplaceAll(p=0.4),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])

transform_replace_all = transforms.Compose([transforms.ToTensor(),
                                RandomSingleColorReplaceAll(p=0.4),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])


transform_random_erase = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])

mnist_train_dataset = datasets.MNIST(root='../data/MNIST', train=True, download=True,
                                     transform=transform)

mnist_train_dataset_replace_all =  datasets.MNIST(root='../data/MNIST', train=True, download=True,
                                     transform=transform_replace_all)

mnist_train_dataset_random_erase =  datasets.MNIST(root='../data/MNIST', train=True, download=True,
                                     transform=transform_random_erase)

mnist_valid_dataset = datasets.MNIST(root='../data/MNIST', train=True, download=True,
                                     transform=transform)
mnist_test_dataset = datasets.MNIST(root='../data/MNIST', train=False, transform=transform)

indices = list(range(len(mnist_train_dataset)))
validation_size = 5000
train_idx, valid_idx = indices[validation_size:], indices[:validation_size]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

mnist_train_loader = DataLoader(
    mnist_train_dataset,
    batch_size=params.batch_size,
    sampler=train_sampler,
    num_workers=params.num_workers
)

mnist_train_loader_replace_all = DataLoader(
    mnist_train_dataset_replace_all,
    batch_size=params.batch_size,
    sampler=train_sampler,
    num_workers=params.num_workers
)

mnist_train_loader_random_erase = DataLoader(
    mnist_train_dataset_random_erase,
    batch_size=params.batch_size,
    sampler=train_sampler,
    num_workers=params.num_workers
)

mnist_valid_loader = DataLoader(
    mnist_valid_dataset,
    batch_size=params.batch_size,
    sampler=train_sampler,
    num_workers=params.num_workers
)

mnist_test_loader = DataLoader(
    mnist_test_dataset,
    batch_size=params.batch_size,
    num_workers=params.num_workers
)


def one_hot_embedding(labels, num_classes=10):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]
