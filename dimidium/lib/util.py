#  /*******************************************************************************
#   * Copyright 2019 -- 2023 IBM Corporation
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

#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Jun 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Utility functions for DOSA/dimidium
#  *
#  *

import re
from enum import Enum
import math
import copy
import numpy as np

from dimidium.lib.units import config_bits_per_byte
from dimidium.lib.dosa_dtype import convert_tvmDtype_to_DosaDtype, get_bitwidth_of_DosaDtype, DosaDtype


# Dosa Return Value
class DosaRv(Enum):
    OK = 0
    ERROR = 1


class OptimizationStrategies(Enum):
    THROUGHPUT = 1
    RESOURCES = 2
    LATENCY = 3
    PERFORMANCE = THROUGHPUT
    DEFAULT = THROUGHPUT


class BrickImplTypes(Enum):
    UNDECIDED = 0
    STREAM = 1
    ENGINE = 2


# based on: https://www.oreilly.com/library/view/python-cookbook/0596001673/ch03s15.html
def multiple_replce(text, repdict):
    # Create a regular expression from all of the dictionary keys
    regex = re.compile("|".join(map(re.escape, repdict.keys())))

    # For each match, look up the corresponding value in the dictionary
    return regex.sub(lambda match: repdict[match.group(0)], text)


# based on: https://stackoverflow.com/questions/65542170/how-to-replace-all-occurence-of-string-in-a-nested-dict
def replace_deep(dicttoreplace, repdict):
    if isinstance(dicttoreplace, str):
        return multiple_replce(dicttoreplace, repdict)
    elif isinstance(dicttoreplace, dict):
        return {multiple_replce(k, repdict): replace_deep(v, repdict) for k, v in dicttoreplace.items()}
    elif isinstance(dicttoreplace, list):
        return [replace_deep(v, repdict) for v in dicttoreplace]
    else:
        # nothing to do?
        return dicttoreplace


def deep_update(dicttoupdate, new_value):
    for e in dicttoupdate:
        if type(dicttoupdate[e]) == dict:
            nv = deep_update(dicttoupdate[e], new_value)
            dicttoupdate[e] = nv
        else:
            dicttoupdate[e] = new_value
    return dicttoupdate


def list_compare(l1, l2):
    l1c = copy.deepcopy(l1)
    l2c = copy.deepcopy(l2)
    l1c.sort()
    l2c.sort()
    return l1 == l2


# Attainable performance
# intensity, peak performance, bandwidth
def rf_attainable_performance(i, P_max, b_s):
    # return np.minimum(np.float64(P_max), np.float64(b_s)*i)
    return min(P_max, b_s * i)


def rf_calc_sweet_spot(oi_list, roof_F, b_s):
    # granularity = oi_list[1] - oi_list[0]
    for oi in oi_list:
        if b_s * oi >= roof_F:
            return oi
    return -1


def dtype_to_bit(dtype):
    # if dtype == 'float32' or 'int32':
    #     return 32
    # if dtype == 'float16' or 'int16':
    #     return 16
    # return 32  # default
    if isinstance(dtype, DosaDtype):
        return get_bitwidth_of_DosaDtype(dtype)
    return get_bitwidth_of_DosaDtype(convert_tvmDtype_to_DosaDtype(dtype))


def dtype_to_size_b(dtype):
    bits = dtype_to_bit(dtype)
    # return math.ceil(bits / config_bits_per_byte)
    return bits / config_bits_per_byte


def bit_to_dtype(bit):
    if bit == 32:
        return 'float32'
    if bit == 16:
        return 'float16'
    return 'int32'  # default


# https://stackoverflow.com/questions/51716916/built-in-module-to-calculate-the-least-common-multiple
def my_lcm(a, b):
    return abs(a * b) // math.gcd(a, b)  # // is floor div


# https://stackoverflow.com/questions/57154745/how-to-find-nearest-divisor-to-given-value-with-modulo-zero
def get_next_larger_dividor(n, near, not_possible_factors=None, max_tries=100):
    if max_tries <= 0:
        return -1
    nn = np.divide(n, np.linspace(1, np.ceil(n / near), int(np.ceil(n / near))))
    bd = int(nn[nn % 1 == 0][-1])
    if not_possible_factors is not None and bd in not_possible_factors:
        return get_next_larger_dividor(n, near+1, not_possible_factors, max_tries=max_tries-1)
    return bd


def bit_width_to_tkeep(bit_w):
    byte_width = math.ceil(bit_w / config_bits_per_byte)
    return byte_width_to_tkeep(byte_width)


def dtype_to_tkeep(dtype):
    byte_width = dtype_to_size_b(dtype)
    return byte_width_to_tkeep(byte_width)


def byte_width_to_tkeep(byte_width):
    tkeep = 0x0
    for i in range(byte_width):
        tkeep = tkeep << 1
        tkeep |= 0b1
    return tkeep

