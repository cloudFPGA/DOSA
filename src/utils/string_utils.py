import functools


def pad_left(string, num_pad, char=' '):
    if string[-1] == '\n':
        string = string[:-1]
    string_split = string.split('\n')
    string_split = list(map(lambda line: num_pad * char + line + '\n', string_split))
    return functools.reduce(lambda l, r: l + r, string_split)
