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

from dimidium.lib.units import config_bits_per_byte


class OptimizationStrategies(Enum):
    PERFORMANCE = 1
    RESOURCES = 2
    LATENCY = 3
    THROUGHPUT = PERFORMANCE
    DEFAULT = PERFORMANCE


def convert_oi_list_for_plot(dpl, default_to_ignore=1.0):
    cmpl_list = []
    uinp_list = []
    detail_list = []
    total_flops = 0
    total_uinp_B = 0
    total_param_B = 0
    for l in dpl:
        e = dpl[l]
        cmpl = e['cmpl']
        if cmpl == default_to_ignore or cmpl == 0:
            continue
        uinp = e['uinp']
        name = e['name']
        layer = e['layer']
        cn = {'name': name + "_" + layer + "_engine", 'oi': cmpl}
        un = {'name': name + "_" + layer + "_stream", 'oi': uinp}
        total_flops += e['flop']
        total_param_B += e['parB']
        total_uinp_B += e['inpB']
        cmpl_list.append(cn)
        uinp_list.append(un)
        detail_list.append(e)
    total = {'flops': total_flops, 'para_B': total_param_B, 'uinp_B': total_uinp_B}
    return cmpl_list, uinp_list, total, detail_list


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


# Attainable performance
# intensity, peak performance, bandwidth
def ap(i, P_max, b_s):
    # return np.minimum(np.float64(P_max), np.float64(b_s)*i)
    return min(P_max, b_s*i)


def dtype_to_bit(dtype):
    if dtype == 'float32' or 'int32':
        return 32
    if dtype == 'float16' or 'int16':
        return 16
    return 32  # default


def dtype_to_size_b(dtype):
    bits = dtype_to_bit(dtype)
    return math.ceil(bits/config_bits_per_byte)


def bit_to_dtype(bit):
    if bit == 32:
        return 'float32'
    if bit == 16:
        return 'float16'
    return 'int32'  # default

