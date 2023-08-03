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
#  *    =============================================
#  *     Created: Jun 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Main file for DOSA/gradatim flow
#  *
#  *

import os
import sys
import json
from enum import Enum

import gradatim.lib.singleton as dosa_singelton
from gradatim.frontend.user_constraints import parse_uc_dict
from gradatim.frontend.model_import import user_import_from_onnx, user_import_from_torchscript
from gradatim.middleend.archGen.archGen import arch_gen
import gradatim.lib.plot_2Droofline as plot_2Droofline
import gradatim.lib.plot_3Droofline as plot_3Droofline
import gradatim.backend.devices.builtin as builtin_devices
from gradatim.backend.operatorSets.osgs import builtin_OSGs, fpga_OSGs
from gradatim.backend.operatorSets.BaseOSG import sort_osg_list, filter_osg_list, get_coverege_multiple_osgs
from gradatim.lib.plot_bandwidth import generate_bandwidth_plt
from gradatim.lib.plot_throughput import generate_throughput_plt_nodes, generate_throughput_plt_bricks
from gradatim.backend.commLibs.commlibs import builtin_comm_libs
from gradatim.backend.commLibs.BaseCommLib import sort_commLib_list

__mandatory_config_keys__ = ['input_latency', 'output_latency', 'dtypes', 'dosa_learning', 'build',
                             # 'engine_saving_threshold', 'generate_testbenchs', 'comm_message_pipeline_store',
                             # 'create_rank_0_for_io', 'insert_debug_cores'
                             'utilization', 'dse']


class DosaModelType(Enum):
    ONNX = 0
    TORCHSCRIPT = 1
    BREVITAS = 2
    UNSUPORTED = 255


def dosa(dosa_config_path, model_type: DosaModelType, model_path: str, const_path: str, global_build_dir: str,
         show_graphics: bool = True, generate_build: bool = True, generate_only_stats: bool = False,
         generate_only_coverage: bool = False, calibration_data: str = None):
    __filedir__ = os.path.dirname(os.path.abspath(__file__))

    with open(dosa_config_path, 'r') as inp:
        dosa_config = json.load(inp)

    for k in __mandatory_config_keys__:
        if k not in dosa_config:
            print("ERROR: Mandatory key {} is missing in the configuration file {}. Stop.".format(k, const_path))
            exit(1)
    dosa_singelton.init_singleton(dosa_config, main_path=__filedir__)
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
    user_constraints, arch_gen_strategy, arch_target_devices, arch_fallback_hw, osg_allowlist = parse_uc_dict(
        const_path,
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

    if model_type == DosaModelType.ONNX:
        print("DOSA: Importing ONNX...")
        mod, params = user_import_from_onnx(model_path, user_constraints, debug_mode)
    elif model_type == DosaModelType.TORCHSCRIPT:
        print("DOSA: Importing TorchScript...")
        mod, params = user_import_from_torchscript(model_path, user_constraints, calibration_data, debug_mode)
    else:
        print(f"ERROR: unsupported model type {model_type}.")
        exit(1)
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
                                                                         show_splits=True, show_labels=True,
                                                                         selected_only=True,
                                                                         print_debug=False)
                new_plt1 = plot_2Droofline.generate_roofline_for_node_plt(nn, archDict['draft'],
                                                                          show_splits=False, show_labels=True,
                                                                          selected_only=True,
                                                                          print_debug=False, iter_based=True)
                new_plt2 = plot_3Droofline.generate_roofline_for_node_plt(nn, archDict['draft'],
                                                                          show_splits=False, show_labels=True,
                                                                          selected_only=True,
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


def print_usage(sys_argv):
    print("USAGE: {} ./path/to/dosa_config.json onnx|torchscript ./path/to/model.file ./path/to/constraint.json "
          "./path/to/build_dir [--calibration-data ./path/to/data.npy] "
          "[--no-roofline|--no-build|--only-stats|--only-coverage]"
          .format(sys_argv[0]))
    exit(1)


def cli():
    if len(sys.argv) < 6 or len(sys.argv) > 9:
        print(f"provided {str(len(sys.argv))} args:" + "\t:\t" + str(sys.argv))
        print_usage(sys.argv)

    # TODO: use argparse
    if (len(sys.argv) == 7 and (sys.argv[6] != '--no-roofline' and sys.argv[6] != '--no-build'
                               and sys.argv[6] != '--only-stats' and sys.argv[6] != '--only-coverage')) or \
            (len(sys.argv) == 9 and (sys.argv[8] != '--no-roofline' and sys.argv[8] != '--no-build'
                               and sys.argv[8] != '--only-stats' and sys.argv[8] != '--only-coverage')) or \
            (len(sys.argv) >= 8 and sys.argv[6][:18] != '--calibration-data'):
        print_usage(sys.argv)

    a_dosa_config_path = sys.argv[1]
    model_type_str = sys.argv[2]
    a_model_type = DosaModelType.UNSUPORTED
    if model_type_str == 'onnx':
        a_model_type = DosaModelType.ONNX
    elif model_type_str == 'torchscript':
        a_model_type = DosaModelType.TORCHSCRIPT
    else:
        print("ERROR: unsupported flow")
        print_usage(sys.argv)

    a_model_path = sys.argv[3]
    a_const_path = sys.argv[4]
    a_global_build_dir = os.path.abspath(sys.argv[5])
    a_show_graphics = True
    a_generate_build = True
    a_generate_only_stats = False  # default is part of build
    a_generate_only_coverage = False
    a_calibration_data_path = None

    if len(sys.argv) == 8 or len(sys.argv) == 9:
        if a_model_type != DosaModelType.TORCHSCRIPT:
            print("ERROR: data calibration only supported for torch script flow")
            exit(1)
        a_calibration_data_path = sys.argv[7]

    if len(sys.argv) == 7 or len(sys.argv) == 9:
        opt_arg_index = 6
        if len(sys.argv) == 9:
            opt_arg_index = 8
        if sys.argv[opt_arg_index] == '--no-roofline':
            a_show_graphics = False
        if sys.argv[opt_arg_index] == '--no-build':
            a_generate_build = False
        if sys.argv[opt_arg_index] == '--only-stats':
            a_show_graphics = False
            a_generate_build = False
            a_generate_only_stats = True
        if sys.argv[opt_arg_index] == '--only-coverage':
            a_show_graphics = False
            a_generate_build = False
            a_generate_only_stats = False
            a_generate_only_coverage = True

    dosa(a_dosa_config_path, a_model_type, a_model_path, a_const_path, a_global_build_dir, a_show_graphics,
         a_generate_build, a_generate_only_stats, a_generate_only_coverage, a_calibration_data_path)


if __name__ == '__main__':
    cli()
