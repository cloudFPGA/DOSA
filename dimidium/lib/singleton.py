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
#  *     Created: Nov 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Singleton object for DOSA ("Quick'n'dirty" way)
#  *

import os
from types import SimpleNamespace
import git

from dimidium.lib.dosa_dtype import DosaDtype, convert_tvmDtype_to_DosaDtype


__filedir__ = os.path.dirname(os.path.abspath(__file__))
is_initiated = False
config = SimpleNamespace()
config.git_version = 'UNKNOWN'
uc = {}


def init_singleton(config_dict, main_path=None):
    global config
    global is_initiated

    config.backend = SimpleNamespace()
    config.backend.input_latency = config_dict['input_latency']
    config.backend.output_latency = config_dict['output_latency']
    config.backend.create_rank_0_for_io = bool(config_dict['build']['create_rank_0_for_io'])
    config.backend.comm_message_pipeline_store = int(config_dict['build']['comm_message_interleaving'])  # to be updated during runtime
    config.backend.comm_message_interleaving = int(config_dict['build']['comm_message_interleaving'])
    config.backend.maximum_pipeline_store_per_node = int(config_dict['build']['maximum_pipeline_store_per_node'])
    config.backend.generate_testbenchs = config_dict['build']['generate_testbenchs']
    config.backend.insert_debug_cores = bool(config_dict['build']['insert_debug_cores'])
    config.backend.tmux_parallel_build = int(config_dict['build']['parallel_builds_tmux'])
    config.backend.clean_build = bool(config_dict['build']['start_from_clean_build'])
    config.backend.comm_message_max_buffer_interleaving = int(config_dict['build']['max_buffer_interleaving'])
    config.backend.allow_multiple_cpu_clients = bool(config_dict['build']['allow_multiple_cpu_clients'])

    config.dtype = SimpleNamespace()
    config.dtype.default_dosa_flops_conv_factor = float(config_dict['dtypes']['default_flops_conv_factor'])
    config.dtype.dosa_flops_base_type = convert_tvmDtype_to_DosaDtype(config_dict['dtypes']['flops_base_type'])
    config.dtype.flops_base_str = config_dict['dtypes']['flops_base_str']
    config.dtype.flops_per_dsp_xilinx_fpgas = float(config_dict['dtypes']['flops_per_dsp_xilinx_fpgas'])
    config.dtype.dsps_per_dosa_flops_xilinx_fpgas = (1 / config.dtype.flops_per_dsp_xilinx_fpgas)
    config.dtype.dosa_flops_explanation_str = 'using {} DSPs per FLOPS'\
        .format(config.dtype.dsps_per_dosa_flops_xilinx_fpgas)

    config.dtype.dosa_kappa = float(config_dict['dosa_learning']['kappa'])
    config.dtype.dosa_lambda = {}
    for k in config_dict['dosa_learning']['lambda']:
        if k == 'fallback':
            config.dtype.dosa_lambda[DosaDtype.UNKNOWN] = float(config_dict['dosa_learning']['lambda'][k])
        else:
            config.dtype.dosa_lambda[convert_tvmDtype_to_DosaDtype(k)] = float(config_dict['dosa_learning']['lambda'][k])

    config.quant = SimpleNamespace()
    config.quant.overwrite_imported_dtypes = False
    config.quant.overwrite_fixed_point_dtypes = False
    config.quant.numbers_already_scaled = True  # TODO: change default to False (if quant module is merged)
    # config.quant.use_extra_accum_dtype = False
    config.quant.activation_dtype = DosaDtype.UNKNOWN
    config.quant.weight_dtype = DosaDtype.UNKNOWN
    config.quant.bias_dtype = DosaDtype.UNKNOWN
    config.quant.fixed_point_fraction_bits = None
    # config.quant.per_layer_dtypes = {}


    config.middleend = SimpleNamespace()
    config.middleend.engine_saving_threshold = float(config_dict['build']['engine_saving_threshold'])

    config.utilization = SimpleNamespace()
    config.utilization.dosa_mu_comp = float(config_dict['dosa_learning']['mu']['compute'])
    config.utilization.dosa_mu_mem = float(config_dict['dosa_learning']['mu']['memory'])
    config.utilization.xilinx_luts_to_dsp_factor = float(config_dict['utilization']['xilinx_luts_to_dsp_factor'])
    config.utilization.xilinx_lutram_to_bram_factor = float(config_dict['utilization']['xilinx_lutram_to_bram_factor'])
    config.utilization.dosa_xi = float(config_dict['utilization']['max_utilization_fpgas'])
    config.utilization.dosa_xi_exception = float(config_dict['utilization']['max_utilization_fpgas']) + \
                                           float(config_dict['utilization']['utilization_exception'])

    config.dse = SimpleNamespace()
    config.dse.allow_throughput_degradation = bool(config_dict['dse']['allow_throughput_degradation'])
    config.dse.allowed_throughput_degradation = 0.0
    if config.dse.allow_throughput_degradation:
        config.dse.allowed_throughput_degradation = float(config_dict['dse']['allowed_throughput_degradation'])
        print("[DOSA:config:INFO] Allowing a degredation of the throughput of {} from the targeted throughput."
              .format(config.dse.allowed_throughput_degradation))

    config.dse.max_vertical_split = 500

    if main_path is not None:
        repo = git.Repo(path=main_path, search_parent_directories=True)
        cur_sha = repo.git.describe()
        config.git_version = cur_sha

    is_initiated = True
    return 0


def add_global_build_dir(abs_path):
    global config
    if config.backend.clean_build:
        os.system("rm -rf {}".format(abs_path))
    else:
        print('[DOSA:build:INFO] Not deleting existing content in output dir.')
    os.system("mkdir -p {}".format(abs_path))
    config.global_build_dir = abs_path
    config.global_report_dir = os.path.abspath('{}/tmp_rpt_dir'.format(abs_path))
    os.system("mkdir -p {}".format(config.global_report_dir))
    os.system("cp {}/../backend/buildTools/templates/dosa_report.py {}/".format(__filedir__, abs_path))
    return 0


def add_user_constraints(uc_dict):
    global uc
    uc = uc_dict
    return 0


