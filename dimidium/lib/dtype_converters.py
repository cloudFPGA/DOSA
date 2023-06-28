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
#  *        Converters for DOSA dtypes
#  *

import dimidium.lib.singleton as dosa_singleton
from dimidium.lib.dosa_dtype import DosaDtype

# config_default_dosa_flops_conv_factor = 1.0
# config_dosa_flops_base_type = DosaDtype.float32
# config_dosa_flops_base_str = 'FLOPS/s for float32'
# config_dosa_flops_per_dsp_xilinx_fpgas = 0.5  # usually, 2 DSPs are required for one float32 multiplication
# config_dsps_per_dosa_flops_xilinx_fpgas = (1 / config_dosa_flops_per_dsp_xilinx_fpgas)
# config_dosa_flops_explanation_str = 'using {} DSPs per FLOPS'.format(config_dsps_per_dosa_flops_xilinx_fpgas)
#
# config_dosa_kappa = 1.0
#
# config_dosa_lambda_dict = {DosaDtype.UNKNOWN: 1.0, DosaDtype.float32: 1.0, DosaDtype.float16: 1.0, DosaDtype.int32: 1.0,
#                            DosaDtype.int16: 1.0, DosaDtype.uint8: 1.0, DosaDtype.double: 1.0}
#


def get_flops_conv_factor(dtype: DosaDtype):
    assert dosa_singleton.is_initiated
    # based on Xilinx resources and @Parker2014
    # ret = config_default_dosa_flops_conv_factor
    ret = dosa_singleton.config.dtype.default_dosa_flops_conv_factor
    if dtype == DosaDtype.float32:
        # requires 2 DSPs
        # return 1.0
        # ret = (2.0 / config_dsps_per_dosa_flops_xilinx_fpgas)
        ret = (2.0 / dosa_singleton.config.dtype.dsps_per_dosa_flops_xilinx_fpgas)
    elif dtype == DosaDtype.float16:
        # also requires 2 DPSs
        # return 1.0
        # ret = (2.0 / config_dsps_per_dosa_flops_xilinx_fpgas)
        ret = (2.0 / dosa_singleton.config.dtype.dsps_per_dosa_flops_xilinx_fpgas)
    elif dtype == DosaDtype.int32:
        # requires 4 DPSs
        # return 2.0
        # ret = (4.0 / config_dsps_per_dosa_flops_xilinx_fpgas)
        ret = (4.0 / dosa_singleton.config.dtype.dsps_per_dosa_flops_xilinx_fpgas)
    elif dtype == DosaDtype.int16:
        # requires 1 DSP
        # return 0.5
        # ret = (1.0 / config_dsps_per_dosa_flops_xilinx_fpgas)
        ret = (1.0 / dosa_singleton.config.dtype.dsps_per_dosa_flops_xilinx_fpgas)
    elif dtype == DosaDtype.uint8:
        # requires 0 or 0.5 DSPs
        # return 0.25  # or 0.2?
        # using 0.3 as middle between 0 and 0.5
        # ret = (0.3 / config_dsps_per_dosa_flops_xilinx_fpgas)
        ret = (0.3 / dosa_singleton.config.dtype.dsps_per_dosa_flops_xilinx_fpgas)
    elif dtype == DosaDtype.int8:
        # requires 0 or 0.5 DSPs
        # return 0.25  # or 0.2?
        # using 0.3 as middle between 0 and 0.5
        # ret = (0.3 / config_dsps_per_dosa_flops_xilinx_fpgas)
        ret = (0.3 / dosa_singleton.config.dtype.dsps_per_dosa_flops_xilinx_fpgas)
    elif dtype == DosaDtype.double:
        # requires 8 DSPs
        # return 4
        # ret = (8.0 / config_dsps_per_dosa_flops_xilinx_fpgas)
        ret = (8.0 / dosa_singleton.config.dtype.dsps_per_dosa_flops_xilinx_fpgas)
    # elif dtype == DosaDtype.UNKNOWN:
    return ret * dosa_singleton.config.dtype.dosa_lambda[dtype]


