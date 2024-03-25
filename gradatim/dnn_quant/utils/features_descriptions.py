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
import functools
from typing import Optional, List
from torch import nn


def pad_left(string, num_pad, char=' '):
    if string[-1] == '\n':
        string = string[:-1]
    string_split = string.split('\n')
    string_split = list(map(lambda line: num_pad * char + line + '\n', string_split))
    return functools.reduce(lambda l, r: l + r, string_split)


def features_descriptions_to_string(descriptions: List[Optional[str]], features: List[nn.Module]) -> str:
    for i, description in enumerate(descriptions):
        if description is None:
            descriptions[i] = '(' + str(i) + '): '
            descriptions[i] += type(features[i]).__name__ + '(ignored in forward pass)\n'
    return functools.reduce(lambda l, r: l + r, descriptions)

