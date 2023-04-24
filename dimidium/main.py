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
from dimidium.frontend.model_import import user_import
from dimidium.middleend.archGen.archGen import arch_gen
import dimidium.lib.plot_2Droofline as plot_2Droofline
import dimidium.lib.plot_3Droofline as plot_3Droofline
import dimidium.backend.devices.builtin as builtin_devices
from dimidium.backend.operatorSets.osgs import builtin_OSGs, fpga_OSGs
from dimidium.backend.operatorSets.BaseOSG import sort_osg_list, filter_osg_list, get_coverege_multiple_osgs
from dimidium.lib.plot_bandwidth import generate_bandwidth_plt
from dimidium.lib.plot_throughput import generate_throughput_plt_nodes, generate_throughput_plt_bricks
from dimidium.backend.commLibs.commlibs import builtin_comm_libs
from dimidium.backend.commLibs.BaseCommLib import sort_commLib_list


__mandatory_config_keys__ = ['input_latency', 'output_latency', 'dtypes', 'dosa_learning', 'build',
                             # 'engine_saving_threshold', 'generate_testbenchs', 'comm_message_pipeline_store',
                             # 'create_rank_0_for_io', 'insert_debug_cores'
                             'utilization', 'dse']


def print_usage(sys_argv):
    print("USAGE: {} ./path/to/dosa_config.json ./path/to/nn.onnx ./path/to/constraint.json ./path/to/build_dir "
          .format(sys_argv[0]) +
          "[--no-roofline|--no-build|--only-stats|--only-coverage]")
    exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 5 or len(sys.argv) > 6:
        print(str(len(sys.argv)) + "\t:\t" + str(sys.argv))
        print_usage(sys.argv)

    # TODO: use argparse
    if len(sys.argv) == 6 and (sys.argv[5] != '--no-roofline' and sys.argv[5] != '--no-build'
        and sys.argv[5] != '--only-stats' and sys.argv[5] != '--only-coverage'):
        print_usage(sys.argv)

    dosa_config_path = sys.argv[1]
    onnx_path = sys.argv[2]
    const_path = sys.argv[3]
    global_build_dir = os.path.abspath(sys.argv[4])
    show_graphics = True
    generate_build = True
    generate_only_stats = False  # default is part of build
    generate_only_coverage = False
    if len(sys.argv) == 6 and sys.argv[5] == '--no-roofline':
        show_graphics = False
    if len(sys.argv) == 6 and sys.argv[5] == '--no-build':
        generate_build = False
    if len(sys.argv) == 6 and sys.argv[5] == '--only-stats':
        show_graphics = False
        generate_build = False
        generate_only_stats = True
    if len(sys.argv) == 6 and sys.argv[5] == '--only-coverage':
        show_graphics = False
        generate_build = False
        generate_only_stats = False
        generate_only_coverage = True

    with open(dosa_config_path, 'r') as inp:
        dosa_config = json.load(inp)

    for k in __mandatory_config_keys__:
        if k not in dosa_config:
            print("ERROR: Mandatory key {} is missing in the configuration file {}. Stop.".format(k, const_path))
            exit(1)
    dosa_singelton.init_singleton(dosa_config, main_path=sys.argv[0])
    dosa_singelton.add_global_build_dir(global_build_dir)

    debug_mode = False

    print("DOSA: Building OSGs, communication and device libraries...")
    available_devices = builtin_devices
    # TODO: extend this object with custom devices
    all_OSGs = builtin_OSGs
    available_OSGs = sort_osg_list(all_OSGs, use_internal_prio=False)
    # TODO: extend this list with custom OSGs here
    print_osg_stats = False
    # init osgs
    prio_int = 0  # get unique internal priorities
    for osg in available_OSGs:
        osg.init(available_devices.classes_dict, prio_int)
        prio_int += 1
        if print_osg_stats:
            osg_cov_stat = osg.get_ir_coverage()
            print(osg_cov_stat)
    if print_osg_stats:
        print(get_coverege_multiple_osgs(fpga_OSGs))
        print(get_coverege_multiple_osgs(fpga_OSGs[:-1]))
    all_commLibs = builtin_comm_libs
    available_comm_libs = sort_commLib_list(all_commLibs, use_internal_prio=False)
    # TODO: extend this list with custom comm libs here
    # init comm libs
    prio_int = 0  # get unique internal priorities
    for comm_lib in available_comm_libs:
        comm_lib.init(available_devices.classes_dict, prio_int)
        prio_int += 1
    print("\t...done.\n")

    print("DOSA: Parsing constraints...")
    user_constraints, arch_gen_strategy, arch_target_devices, arch_fallback_hw, osg_allowlist = parse_uc_dict(const_path,
                                                                                               available_devices)
    dosa_singelton.add_user_constraints(user_constraints)

    if osg_allowlist is not None:
        available_OSGs = filter_osg_list(available_OSGs, osg_allowlist)

    target_sps = user_constraints['target_sps']  # in #/s
    target_latency = user_constraints['target_latency']  # in s per sample
    used_batch = user_constraints['used_batch_n']
    target_resource_budget = user_constraints['target_resource_budget']  # in number of nodes
    used_name = user_constraints['name']
    used_in_size_t = user_constraints['used_input_size_t']
    used_sample_size = user_constraints['used_sample_size']
    print("\t...done.\n")

    print("DOSA: Importing ONNX...")
    mod, params = user_import(onnx_path, user_constraints, debug_mode)
    print("\t...done.\n")

    # TODO: remove temporary guards
    assert (arch_target_devices[0] == builtin_devices.cF_FMKU60_Themisto_1) \
           or (arch_target_devices[0] == builtin_devices.cF_Infinity_1)
    assert len(arch_target_devices) == 1
    print("DOSA: Generating high-level architecture...")
    archDict = arch_gen(mod, params, used_name, arch_gen_strategy, available_OSGs, available_devices,
                        available_comm_libs, used_batch, used_sample_size, target_sps, target_latency,
                        target_resource_budget, arch_target_devices, arch_fallback_hw, debug=debug_mode, profiling=True,
                        verbose=True, generate_build=generate_build, generate_only_stats=generate_only_stats,
                        write_only_osg_coverage=generate_only_coverage)
    print("\t...done.\n")

    all_plots = True
    if archDict['draft'].nid_cnt > 10:
        all_plots = False
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
        if all_plots:
            plt2 = plot_2Droofline.generate_roofline_plt(archDict['draft'], show_splits=False, show_labels=True,
                                                         show_ops=True, selected_only=False, print_debug=False)
        plt22 = plot_2Droofline.generate_roofline_plt(archDict['draft'], show_splits=False, show_labels=True,
                                                     show_ops=True, selected_only=True, print_debug=False)
        plt23 = plot_2Droofline.generate_roofline_plt(archDict['draft'], show_splits=False, show_labels=True,
                                                      show_ops=True, selected_only=True, print_debug=False,
                                                      iter_based=True)
        plt8 = plot_3Droofline.generate_roofline_plt(archDict['draft'], show_splits=False, show_labels=True,
                                                     print_debug=False)
        if all_plots:
            plt_nodes = []
            for nn in archDict['draft'].node_iter_gen():
                if nn.skip_in_roofline:
                    continue
                new_plt = plot_2Droofline.generate_roofline_for_node_plt(nn, archDict['draft'],
                                                                         show_splits=True, show_labels=True, selected_only=True,
                                                                         print_debug=False)
                new_plt1 = plot_2Droofline.generate_roofline_for_node_plt(nn, archDict['draft'],
                                                                         show_splits=False, show_labels=True, selected_only=True,
                                                                         print_debug=False, iter_based=True)
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
        plt8 = generate_throughput_plt_bricks(archDict['draft'])
        plt9 = generate_throughput_plt_nodes(archDict['draft'])
        last_plt = plt9

        plot_2Droofline.show_roofline_plt(last_plt, waiting=True)
        print("\t...done.\n")

    print("\nDOSA finished successfully.\n")


