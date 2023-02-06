from torchvision import datasets


def get_dataset(data_dir, dataset, test, transform):
    return datasets[dataset.lower()](
        root=data_dir,
        train=not test,
        download=True,
        transform=transform
    )


datasets = {
    'cifar10': datasets.CIFAR10,
    'cifar100': datasets.CIFAR100,
    'mnist': datasets.MNIST,
}
