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

