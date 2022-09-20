import numpy as np
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler

from .transforms import transforms
from .datasets import get_dataset


def data_loader(data_dir,
                dataset,
                batch_size,
                num_elements=None,
                valid_size=0.01,
                shuffle=True,
                test=False
                ):
    dataset = dataset.lower()
    suffix = '_test' if test else '_train'
    transform = transforms[dataset + suffix]
    data = get_dataset(data_dir, dataset, test, transform)

    num_elements = num_elements if num_elements is not None else len(data)
    indices = list(range(num_elements))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    if test:
        return torch.utils.data.DataLoader(
            data, batch_size=batch_size, sampler=SubsetRandomSampler(indices)
        )

    split = int(np.floor(valid_size * num_elements))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx)
    )

    valid_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, sampler=SubsetRandomSampler(valid_idx)
    )

    return train_loader, valid_loader
