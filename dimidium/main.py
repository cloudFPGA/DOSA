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

from dimidium.frontend.user_constraints import parse_uc_dict
from dimidium.frontend.model_import import onnx_import, tvm_optimization_pass
from dimidium.middleend.archGen import arch_gen
import dimidium.lib.plot_roofline as plot_roofline
import dimidium.backend.devices as dosa_devices
from dimidium.lib.util import config_bits_per_byte

__mandatory_config_keys__ = ['input_latency', 'output_latency']


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
    debug_mode = True

    print("DOSA: Parsing constraints...")
    user_constraints, arch_gen_strategy, arch_target_devices, arch_fallback_hw = parse_uc_dict(const_path)

    target_sps = user_constraints['target_sps']  # in #/s
    target_latency = user_constraints['target_latency']  # in s per sample
    used_batch = user_constraints['used_batch_n']
    target_resource_budget = user_constraints['target_resource_budget']  # in number of nodes
    used_name = user_constraints['name']
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
    print("\t...done.\n")

    print("DOSA: Importing ONNX...")
    mod_i, params_i = onnx_import(onnx_path, user_constraints['shape_dict'])
    print("\t...done.\n")

    print("DOSA: Executing TVM optimization passes...")
    mod, params = tvm_optimization_pass(mod_i, params_i, debug=debug_mode)
    print("\t...done.\n")

    # TODO: remove temporary guards
    assert arch_target_devices[0] == dosa_devices.cF_FMKU60_Themisto_1
    assert len(arch_target_devices) == 1
    print("DOSA: Generating high-level architecture...")
    archDict = arch_gen(mod, params, used_name, arch_gen_strategy, used_batch, used_sample_size, target_sps,
                        target_latency,
                        target_resource_budget, arch_target_devices, arch_fallback_hw, debug=debug_mode)
    print("\t...done.\n")

    print("DOSA: Generating and showing roofline...")
    plt = plot_roofline.generate_roofline_plt_old(archDict['base_dpl'], target_sps, used_batch,
                                                  used_name + " (basis)",
                                                  arch_target_devices[0].get_performance_dict(),
                                                  arch_target_devices[0].get_roofline_dict(),
                                                  show_splits=True, show_labels=True)
    # plt2 = plot_roofline.generate_roofline_plt_old(archDict['fused_view'], target_sps, used_batch,
    #                                                used_name + " (optimized)",
    #                                                arch_target_devices[0].get_performance_dict(),
    #                                                arch_target_devices[0].get_roofline_dict(),
    #                                                show_splits=True, show_labels=True)
    plt2 = plot_roofline.generate_roofline_plt(archDict['draft'], show_splits=True, show_labels=True)
    plt_nodes = []
    for nn in archDict['draft'].node_iter_gen():
        new_plt = plot_roofline.generate_roofline_for_node_plt(nn, archDict['draft'],
                                                               show_splits=True, show_labels=True, selected_only=True)
        plt_nodes.append(new_plt)
    last_plt = plt_nodes[-1]
    if debug_mode:
        plt3 = plot_roofline.generate_roofline_plt(archDict['debug_obj']['other_opts'][0])
        plt4 = plot_roofline.generate_roofline_plt(archDict['debug_obj']['other_opts'][1])
        plt5 = plot_roofline.generate_roofline_plt(archDict['debug_obj']['other_opts'][2])
        last_plt = plt5

    # plot_roofline.show_roofline_plt(plt, blocking=False) not necessary...
    plot_roofline.show_roofline_plt(last_plt)
    print("\t...done.\n")

    print("\nDOSA finished successfully.\n")
