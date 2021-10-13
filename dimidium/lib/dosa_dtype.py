#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Enum for different data types used in DOSA
#  *
#  *

from enum import Enum


class DosaDtype(Enum):
    UNKNOWN = 0
    float32 = 1
    float16 = 2
    int32   = 3
    int16   = 4
    uint8   = 5
    double  = 6


def convert_tvmDtype_to_DosaDtype(dtype):
    if dtype == 'float32':
        return DosaDtype.float32
    elif dtype == 'float16':
        return DosaDtype.float16
    elif dtype == 'int32':
        return DosaDtype.int32
    elif dtype == 'int16':
        return DosaDtype.int16
    elif dtype == 'uint8':
        # TODO: catch also uint below 8 here?
        return DosaDtype.uint8
    return DosaDtype.UNKNOWN
    # return DosaDtype.float32  # default?


config_default_dosa_flops_conv_factor = 1.0
config_dosa_flops_base_type = DosaDtype.float32
config_dosa_flops_base_str = 'FLOPS/s for float32'
config_dosa_flops_per_dsp_xilinx_fpgas = 0.5  # usually, 2 DSPs are required for one float32 multiplication
config_dsps_per_dosa_flops_xilinx_fpgas = (1 / config_dosa_flops_per_dsp_xilinx_fpgas)
config_dosa_flops_explanation_str = 'using {} DSPs per FLOPS'.format(config_dsps_per_dosa_flops_xilinx_fpgas)


def get_flops_conv_factor(dtype: DosaDtype):
    # based on Xilinx resources and @Parker2014
    if dtype == DosaDtype.float32:
        # requires 2 DSPs
        # return 1.0
        return 2.0 / config_dsps_per_dosa_flops_xilinx_fpgas
    elif dtype == DosaDtype.float16:
        # also requires 2 DPSs
        # return 1.0
        return 2.0 / config_dsps_per_dosa_flops_xilinx_fpgas
    elif dtype == DosaDtype.int32:
        # requires 4 DPSs
        # return 2.0
        return 4.0 / config_dsps_per_dosa_flops_xilinx_fpgas
    elif dtype == DosaDtype.int16:
        # requires 1 DSP
        # return 0.5
        return 1.0 / config_dsps_per_dosa_flops_xilinx_fpgas
    elif dtype == DosaDtype.uint8:
        # requires 0 or 0.5 DSPs
        # return 0.25  # or 0.2?
        # using 0.3 as middle between 0 and 0.5
        return 0.3 / config_dsps_per_dosa_flops_xilinx_fpgas
    elif dtype == DosaDtype.double:
        # requires 8 DSPs
        # return 4
        return 8.0 / config_dsps_per_dosa_flops_xilinx_fpgas
    # if dtype == DosaDtype.UNKNOWN:
    return config_default_dosa_flops_conv_factor


