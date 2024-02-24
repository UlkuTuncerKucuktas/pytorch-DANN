import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
import torch
import params
from transformations import *

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


transformations = {
    "RandomHorizontalFlip": transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        base_transform
    ]),
    "RandomRotation": transforms.Compose([
        transforms.RandomRotation(degrees=15,fill=0),
        base_transform
    ]),
    "ColorJitter": transforms.Compose([
        transforms.ColorJitter(),
        base_transform
    ]),
    "RandomErasing": transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomErasing(value=255,p=0.5,scale=(0.02, 0.10)),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    "GaussianBlur": transforms.Compose([
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        base_transform
    ]),
    "RandomVerticalFlip": transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        base_transform
    ]),
    "RandomInvert": transforms.Compose([
        transforms.RandomInvert(p=0.5),
        base_transform
    ]),
    "RandomAdjustSharpness": transforms.Compose([
        transforms.RandomAdjustSharpness(sharpness_factor=2,p=1),
        base_transform
    ]),
    "RandomAutocontrast": transforms.Compose([
        transforms.RandomAutocontrast(p=0.5),
        base_transform
    ]),
    "RandomSingleColorReplaceBlack": transforms.Compose([
        RandomSingleColorReplaceBlack(p=0.5),
        base_transform
    ]),
    "RandomSingleColorReplaceNonBlack": transforms.Compose([
        RandomSingleColorReplaceNonBlack(p=0.5),
        base_transform
    ]),
    "RandomSingleColorReplaceAll": transforms.Compose([
        RandomSingleColorReplaceAll(p=0.5),
        base_transform
    ]),
    "RandomColorsReplaceBlack": transforms.Compose([
        RandomColorsReplaceBlack(p=0.5),
        base_transform
    ]),
    "RandomColorsReplaceNonBlack": transforms.Compose([
        RandomColorsReplaceNonBlack(p=0.5),
        base_transform
    ]),
    "Identity": base_transform
}




aug_datasets_dict = {name: datasets.MNIST(root='../data/MNIST', train=True, download=True, transform=transform)
                 for name, transform in transformations.items()}



mnist_train_dataset = datasets.MNIST(root='../data/MNIST', train=True, download=True,
                                     transform=base_transform)



mnist_valid_dataset = datasets.MNIST(root='../data/MNIST', train=True, download=True,
                                     transform=base_transform)
                                     
mnist_test_dataset = datasets.MNIST(root='../data/MNIST', train=False, transform=base_transform)




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

dataloader_dict = {
    name: DataLoader(
        dataset,
        batch_size=params.batch_size,
        sampler=train_sampler, 
        num_workers=params.num_workers
    )
    for name, dataset in aug_datasets_dict.items()
}




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
