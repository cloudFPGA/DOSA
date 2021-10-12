#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *       Library for parsing and checking user constraints
#  *
#  *

import math
import json

from dimidium.lib.util import OptimizationStrategies, config_bits_per_byte

__mandatory_user_keys__ = ['shape_dict', 'used_batch_n', 'name', 'target_sps', 'targeted_hw',
                           'target_resource_budget', 'arch_gen_strategy', 'fallback_hw', 'used_input_size_t',
                           'target_latency']
__arch_gen_strategies__ = ['performance', 'resources', 'default', 'latency', 'throughput']
# __valid_fallback_hws__ = ['None']
# __valid_fallback_hws__.extend(dosa_devices.fallback_hw)


def parse_uc_dict(path, dosa_devices):
    with open(path, 'r') as inp:
        user_constraints = json.load(inp)

    for k in __mandatory_user_keys__:
        if k not in user_constraints:
            print("ERROR: Mandatory key {} is missing in the constraints file {}. Stop.".format(k, path))
            exit(1)

    arch_target_devices = []
    for td in user_constraints['targeted_hw']:
        if td not in dosa_devices.types_str:
            print("ERROR: Target hardware {} is not supported. Stop.".format(td))
            exit(1)
        else:
            arch_target_devices.append(dosa_devices.types_dict[td])

    uags = user_constraints['arch_gen_strategy']
    if uags not in __arch_gen_strategies__:
        print("ERROR: Architecture optimization strategy {} is not supported. Stop.".format(uags))
        exit(1)

    arch_gen_strategy = OptimizationStrategies.DEFAULT
    if uags == 'performance':
        arch_gen_strategy = OptimizationStrategies.PERFORMANCE
    elif uags == 'resources':
        arch_gen_strategy = OptimizationStrategies.RESOURCES
    elif uags == 'latency':
        arch_gen_strategy = OptimizationStrategies.LATENCY
    elif uags == 'throughput':
        arch_gen_strategy = OptimizationStrategies.THROUGHPUT
    else:
        arch_gen_strategy = OptimizationStrategies.DEFAULT

    arch_fallback_hw = None
    if type(user_constraints['fallback_hw']) is list:
        for fhw in user_constraints['fallback_hw']:
            if fhw not in dosa_devices.types_str:
                print("ERROR: Fallback hardware {} is not supported. Stop.".format(fhw))
                exit(1)
        arch_fallback_hw = user_constraints['fallback_hw']
    else:
        if user_constraints['fallback_hw'] != "None":
            print("ERROR: Fallback hardware {} is not supported. Stop."
                  .format(user_constraints['fallback_hw']))
            exit(1)
        # arch_fallback_hw stays None

    used_batch = user_constraints['used_batch_n']
    used_in_size_t = user_constraints['used_input_size_t']
    sample_size_bit = used_in_size_t
    for inp_k in user_constraints['shape_dict']:
        inp_v = user_constraints['shape_dict'][inp_k]
        total_d = len(inp_v)
        if used_batch != inp_v[0]:
            print("ERROR: Batch sizes in constraint file contradict each other ({} != {}). Stop."
                  .format(used_batch, inp_v[0]))
            exit(1)
        for i in range(1, total_d):  # exclude batch size
            d = inp_v[i]
            sample_size_bit *= d
    used_sample_size = math.ceil(sample_size_bit / config_bits_per_byte)
    user_constraints['used_sample_size'] = used_sample_size

    return user_constraints, arch_gen_strategy, arch_target_devices, arch_fallback_hw

