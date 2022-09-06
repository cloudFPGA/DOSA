import os
import PIL
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def data_loader(data_dir,
                batch_size,
                num_elements=None,
                valid_size=0.1,
                shuffle=True,
                test=False):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=not test,
        download=True,
        transform=transform,
    )

    if test:
        num_test = num_elements if num_elements is not None else len(dataset)
        indices = list(range(num_test))

        if shuffle:
            np.random.seed(42)
            np.random.shuffle(indices)

        test_sampler = SubsetRandomSampler(indices)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=test_sampler
        )
        return data_loader

    num_train = num_elements if num_elements is not None else len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler
    )

    return train_loader, valid_loader


class ImageNetSamplesDataset(Dataset):
    def __init__(self, root, transform=None):
        # self.dataset_dir = os.path.join(os.path.join(os.getcwd(), root), 'imagenet-sample')
        self.dataset_dir = os.path.join(root, 'imagenet-sample')
        self.all_filenames = os.listdir(self.dataset_dir)
        self.all_labels = np.arange(0, 1000)
        self.transform = transform

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        selected_filename = self.all_filenames[idx]
        image_pil = PIL.Image.open(os.path.join(self.dataset_dir, selected_filename)).convert('RGB')
        image = self.transform(image_pil)
        label = idx
        sample = (image, label)
        return sample


def imageNet_test_data_loader(data_dir, batch_size):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dataset = ImageNetSamplesDataset(root=data_dir, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size)

