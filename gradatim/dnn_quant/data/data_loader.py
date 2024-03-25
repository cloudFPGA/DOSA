#  /*******************************************************************************
#   * Copyright 2022 -- 2024 IBM Corporation
#   *
#   * Licensed under the Apache License, Version 2.0 (the "License");
#   * you may not use this file except in compliance with the License.
#   * You may obtain a copy of the License at
#   *
#   *     http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS,
#   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   * See the License for the specific language governing permissions and
#   * limitations under the License.
#  *******************************************************************************/
#
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
                test=False,
                seed=0
                ):
    dataset = dataset.lower()
    suffix = '_test' if test else '_train'
    transform = transforms[dataset + suffix]
    data = get_dataset(data_dir, dataset, test, transform)

    num_elements = num_elements if num_elements is not None else len(data)
    indices = list(range(num_elements))

    if shuffle:
        np.random.seed(seed)
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
