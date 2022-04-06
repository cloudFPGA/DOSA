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
from dimidium.lib.dosa_dtype import DosaDtype, convert_tvmDtype_to_DosaDtype


__filedir__ = os.path.dirname(os.path.abspath(__file__))
is_initiated = False
config = SimpleNamespace()


def init_singleton(config_dict):
    global config
    global is_initiated

    config.backend = SimpleNamespace()
    config.backend.input_latency = config_dict['input_latency']
    config.backend.output_latency = config_dict['output_latency']
    config.backend.create_rank_0_for_io = bool(config_dict['build']['create_rank_0_for_io'])
    config.backend.comm_message_interleaving = int(config_dict['build']['comm_message_interleaving'])
    config.backend.generate_testbenchs = config_dict['build']['generate_testbenchs']
    config.backend.insert_debug_cores = bool(config_dict['build']['insert_debug_cores'])
    config.backend.tmux_parallel_build = int(config_dict['build']['parallel_builds_tmux'])
    config.backend.clean_build = bool(config_dict['build']['start_from_clean_build'])

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

