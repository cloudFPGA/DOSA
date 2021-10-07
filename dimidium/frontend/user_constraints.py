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

import json

import dimidium.backend.devices as dosa_devices
from dimidium.lib.util import OptimizationStrategies

__mandatory_user_keys__ = ['shape_dict', 'used_batch_n', 'name', 'target_sps', 'target_hw',
                           'target_resource_budget', 'arch_gen_strategy', 'fallback_hw', 'used_input_size_t',
                           'target_latency']
__arch_gen_strategies__ = ['performance', 'resources', 'default', 'latency', 'throughput']
__valid_fallback_hws__ = ['None']
__valid_fallback_hws__.extend(dosa_devices.fallback_hw)


def parse_uc_dict(path):
    with open(path, 'r') as inp:
        user_constraints = json.load(inp)

    for k in __mandatory_user_keys__:
        if k not in user_constraints:
            print("ERROR: Mandatory key {} is missing in the constraints file {}. Stop.".format(k, const_path))
            exit(1)

    arch_target_devices = []
    for td in user_constraints['target_hw']:
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
            if fhw not in dosa_devices.fallback_hw:
                print("ERROR: Fallback hardware {} is not supported in list of fallback_hw. Stop.".format(fhw))
                exit(1)
        arch_fallback_hw = user_constraints['fallback_hw']
    else:
        if user_constraints['fallback_hw'] != "None":
            print("ERROR: Fallback hardware {} is not supported in non-list of fallback_hw. Stop."
                  .format(user_constraints['fallback_hw']))
            exit(1)
        # arch_fallback_hw stays None

    return user_constraints, arch_gen_strategy, arch_target_devices, arch_fallback_hw

