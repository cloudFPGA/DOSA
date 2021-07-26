#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Jun 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Main file for DOSA/dimidium flow
#  *
#  *

import sys
import json
import math
import numpy as np
import onnx
import tvm
import tvm.relay as relay

from dimidium.lib.archGen import arch_gen
import dimidium.lib.plot_roofline as plot_roofline
import dimidium.lib.devices as dosa_devices
from dimidium.lib.util import OptimizationStrategies
from dimidium.lib.util import config_bits_per_byte

__mandatory_config_keys__ = ['input_latency', 'output_latency']
__mandatory_user_keys__ = ['shape_dict', 'used_batch_n', 'name', 'target_sps', 'target_hw',
                           'target_resource_budget', 'arch_gen_strategy', 'fallback_hw', 'used_input_size_t',
                           'target_latency']
__arch_gen_strategies__ = ['performance', 'resources', 'default', 'latency', 'throughput']
__valid_fallback_hws__ = ['None']
__valid_fallback_hws__.extend(dosa_devices.fallback_hw)


def onnx_import(onnx_path, shape_dict, debug=False):
    onnx_model = onnx.load(onnx_path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
    if debug:
        print(mod.astext(show_meta_data=False))
    return mod, params


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("USAGE: {} ./path/to/dosa_config.json ./path/to/nn.onnx ./path/to/constraint.json".format(sys.argv[0]))
        exit(1)

    dosa_config_path = sys.argv[1]
    onnx_path = sys.argv[2]
    const_path = sys.argv[3]

    with open(dosa_config_path, 'r') as inp:
        dosa_config = json.load(inp)

    for k in __mandatory_config_keys__:
        if k not in dosa_config:
            print("ERROR: Mandatory key {} is missing in the configuration file {}. Stop.".format(k, const_path))
            exit(1)

    with open(const_path, 'r') as inp:
        user_constraints = json.load(inp)

    for k in __mandatory_user_keys__:
        if k not in user_constraints:
            print("ERROR: Mandatory key {} is missing in the constraints file {}. Stop.".format(k, const_path))
            exit(1)

    arch_target_devices = []
    for td in user_constraints['target_hw']:
        if td not in dosa_devices.types:
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

    print("DOSA: Importing ONNX...")
    mod, params = onnx_import(onnx_path, user_constraints['shape_dict'])
    target_sps = user_constraints['target_sps']
    target_latency = user_constraints['target_latency']
    used_batch = user_constraints['used_batch_n']
    target_resource_budget = user_constraints['resource_budget']
    used_name = user_constraints['name']
    used_in_size_t = user_constraints['used_input_size_t']
    sample_size_bit = used_in_size_t
    for inp_k in user_constraints['shape_dict']:
        inp_v = user_constraints['shape_dict'][inp_k]
        total_d = len(inp_v)
        for i in range(1, total_d):  # exclude batch size
            d = inp_v[i]
            sample_size_bit *= d
    used_sample_size = math.ceil(sample_size_bit/config_bits_per_byte)
    print("\t...done.\n")

    # TODO: remove temporary guards
    assert arch_target_devices[0] == dosa_devices.cF_FMKU60_Themisto_1
    assert len(arch_target_devices) == 1
    print("DOSA: Generating high-level architecture...")
    archDict = arch_gen(mod, params, debug=True)
    print("\t...done.\n")

    print("DOSA: Generating and showing roofline...")
    plt = plot_roofline.generate_roofline_plt(archDict['dpl'], target_sps, used_batch, used_name,
                                              arch_target_devices[0].get_performance_dict(),
                                              arch_target_devices[0].get_roofline_dict(),
                                              show_splits=True, show_labels=True)
    plt2 = plot_roofline.generate_roofline_plt(archDict['fused_view'], target_sps, used_batch,
                                               used_name + " (optimized)",
                                               arch_target_devices[0].get_performance_dict(),
                                               arch_target_devices[0].get_roofline_dict(),
                                               show_splits=True, show_labels=True)
    # plot_roofline.show_roofline_plt(plt, blocking=False) not necessary...
    plot_roofline.show_roofline_plt(plt2)
    print("\t...done.\n")

    print("\nDOSA finished successfully.\n")
