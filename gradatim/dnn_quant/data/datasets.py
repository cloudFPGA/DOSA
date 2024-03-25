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
