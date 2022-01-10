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

import os
import sys
import json
import math

import dimidium.lib.singleton as dosa_singelton
from dimidium.frontend.user_constraints import parse_uc_dict
from dimidium.frontend.model_import import onnx_import, tvm_optimization_pass
from dimidium.middleend.archGen.archGen import arch_gen
import dimidium.lib.plot_2Droofline as plot_2Droofline
import dimidium.lib.plot_3Droofline as plot_3Droofline
import dimidium.backend.devices.builtin as builtin_devices
from dimidium.backend.operatorSets.osgs import builtin_OSGs
from dimidium.backend.operatorSets.BaseOSG import sort_osg_list
from dimidium.lib.plot_bandwidth import generate_bandwidth_plt


__mandatory_config_keys__ = ['input_latency', 'output_latency', 'dtypes', 'dosa_learning', 'engine_saving_threshold',
                             'generate_testbenchs', 'utilization']


def print_usage(sys_argv):
    print("USAGE: {} ./path/to/dosa_config.json ./path/to/nn.onnx ./path/to/constraint.json ./path/to/build_dir "
          .format(sys.argv[0]) +
          "[--no-roofline]")
    exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 5 or len(sys.argv) > 6:
        print_usage(sys.argv)

    if len(sys.argv) == 6 and sys.argv[5] != '--no-roofline':
        print_usage(sys.argv)

    dosa_config_path = sys.argv[1]
    onnx_path = sys.argv[2]
    const_path = sys.argv[3]
    global_build_dir = os.path.abspath(sys.argv[4])
    show_graphics = True
    if len(sys.argv) == 6 and sys.argv[5] == '--no-roofline':
        show_graphics = False

    with open(dosa_config_path, 'r') as inp:
        dosa_config = json.load(inp)

    for k in __mandatory_config_keys__:
        if k not in dosa_config:
            print("ERROR: Mandatory key {} is missing in the configuration file {}. Stop.".format(k, const_path))
            exit(1)
    dosa_singelton.init_singleton(dosa_config)
    dosa_singelton.add_global_build_dir(global_build_dir)

    debug_mode = False

    print("DOSA: Building OSGs and device library...")
    available_devices = builtin_devices
    # TODO: extend this object with custom devices
    all_OSGs = builtin_OSGs
    available_OSGs = sort_osg_list(all_OSGs, use_internal_prio=False)
    # TODO: extend this list with custom OSGs here
    # init osgs
    prio_int = 0  # get unique internal priorities
    for osg in available_OSGs:
        osg.init(available_devices.classes_dict, prio_int)
        prio_int += 1
    print("\t...done.\n")

    print("DOSA: Parsing constraints...")
    user_constraints, arch_gen_strategy, arch_target_devices, arch_fallback_hw = parse_uc_dict(const_path,
                                                                                               available_devices)

    target_sps = user_constraints['target_sps']  # in #/s
    target_latency = user_constraints['target_latency']  # in s per sample
    used_batch = user_constraints['used_batch_n']
    target_resource_budget = user_constraints['target_resource_budget']  # in number of nodes
    used_name = user_constraints['name']
    used_in_size_t = user_constraints['used_input_size_t']
    used_sample_size = user_constraints['used_sample_size']
    print("\t...done.\n")

    print("DOSA: Importing ONNX...")
    mod_i, params_i = onnx_import(onnx_path, user_constraints['shape_dict'])
    print("\t...done.\n")

    print("DOSA: Executing TVM optimization passes...")
    mod, params = tvm_optimization_pass(mod_i, params_i, debug=debug_mode)
    print("\t...done.\n")

    # TODO: remove temporary guards
    assert arch_target_devices[0] == builtin_devices.cF_FMKU60_Themisto_1
    assert len(arch_target_devices) == 1
    print("DOSA: Generating high-level architecture...")
    archDict = arch_gen(mod, params, used_name, arch_gen_strategy, available_OSGs, available_devices,
                        used_batch, used_sample_size, target_sps, target_latency, target_resource_budget,
                        arch_target_devices, arch_fallback_hw, debug=debug_mode, profiling=True, verbose=True)
    print("\t...done.\n")

    if show_graphics:
        print("DOSA: Generating and showing roofline...")
        plt = plot_2Droofline.generate_roofline_plt_old(archDict['base_dpl'], target_sps, used_batch,
                                                        used_name + " (basis)",
                                                        arch_target_devices[0].get_performance_dict(),
                                                        arch_target_devices[0].get_roofline_dict(),
                                                        show_splits=False, show_labels=True, print_debug=False)
        # plt2 = plot_2Droofline.generate_roofline_plt_old(archDict['fused_view'], target_sps, used_batch,
        #                                                used_name + " (optimized)",
        #                                                arch_target_devices[0].get_performance_dict(),
        #                                                arch_target_devices[0].get_roofline_dict(),
        #                                                show_splits=True, show_labels=True)
        plt2 = plot_2Droofline.generate_roofline_plt(archDict['draft'], show_splits=False, show_labels=True,
                                                     show_ops=True, selected_only=False, print_debug=False)
        plt22 = plot_2Droofline.generate_roofline_plt(archDict['draft'], show_splits=False, show_labels=True,
                                                     show_ops=True, selected_only=True, print_debug=False)
        plt8 = plot_3Droofline.generate_roofline_plt(archDict['draft'], show_splits=False, show_labels=True,
                                                     print_debug=False)
        plt_nodes = []
        for nn in archDict['draft'].node_iter_gen():
            new_plt = plot_2Droofline.generate_roofline_for_node_plt(nn, archDict['draft'],
                                                                     show_splits=True, show_labels=True, selected_only=True,
                                                                     print_debug=False)
            new_plt2 = plot_3Droofline.generate_roofline_for_node_plt(nn, archDict['draft'],
                                                                     show_splits=False, show_labels=True, selected_only=True,
                                                                     print_debug=False)
            plt_nodes.append(new_plt)
        last_plt = plt_nodes[-1]
        if debug_mode:
            plt3 = plot_2Droofline.generate_roofline_plt(archDict['debug_obj']['other_opts'][0])
            plt4 = plot_2Droofline.generate_roofline_plt(archDict['debug_obj']['other_opts'][1])
            plt5 = plot_2Droofline.generate_roofline_plt(archDict['debug_obj']['other_opts'][2])
            last_plt = plt5
        plt7 = generate_bandwidth_plt(archDict['draft'])
        last_plt = plt7

        plot_2Droofline.show_roofline_plt(last_plt, waiting=True)
        print("\t...done.\n")

    print("\nDOSA finished successfully.\n")


