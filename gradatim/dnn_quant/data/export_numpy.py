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
import torch


def export_data_as_npz(path, data_loader, num_batches=None, feature_transform=None, label_transform=None,
                       dtype='int8', seed=None):
    np_features = np.empty(0)
    np_labels = np.empty(0)

    num_batches = num_batches if num_batches is not None else len(data_loader)

    if seed is not None:
        torch.manual_seed(seed)
    batch_id = 0
    for features, labels in data_loader:
        if batch_id == num_batches:
            break
        batch_id += 1

        features = feature_transform(features).numpy() if feature_transform is not None else features.numpy()
        labels = label_transform(labels).numpy() if label_transform is not None else labels.numpy()
        np_features = np.concatenate((np_features, features)) if np_features.shape[0] else features
        np_labels = np.concatenate((np_labels, labels)) if np_labels.shape[0] else labels

    np_features = np_features.astype(dtype)
    np.savez(path, features=np_features, labels=np_labels)


def export_data_as_numpy(path, data, data_transform=None, dtype='int8'):
    data = data_transform(data).numpy() if data_transform is not None else data.numpy()
    data = data.astype(dtype)
    np.save(path, data)
