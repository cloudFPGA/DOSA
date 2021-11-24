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

is_initiated = False
config = SimpleNamespace()


def init_singleton(config_dict):
    global config
    global is_initiated

    config.backend = SimpleNamespace()
    config.backend.input_latency = config_dict['input_latency']
    config.backend.output_latency = config_dict['output_latency']
    config.backend.reserve_rank_0_for_io = config_dict['reserve_rank_0_for_io']

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
    config.middleend.engine_saving_threshold = float(config_dict['engine_saving_threshold'])

    is_initiated = True
    return 0


def add_global_build_dir(abs_path):
    global config
    os.system("mkdir -p {}".format(abs_path))
    config.global_build_dir = abs_path
    return 0

