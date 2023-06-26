#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Operator Set Generator (OSG) library for DOSA arch gen
#  *
#  *

from dimidium.backend.operatorSets.relay_ops import op as relay_op_list
from dimidium.backend.operatorSets.hls4mlOSG import Hls4mlOSG
from dimidium.backend.operatorSets.TvmCpuOsg import TvmCpuOsg
from dimidium.backend.operatorSets.Haddoc2OSG import vhdl4cnnOSG
from dimidium.backend.operatorSets.TipsOSG import TipsOSG
from dimidium.backend.operatorSets.VtaOSG import VtaOSG


# add all available OSGs here
osg_hls4ml = Hls4mlOSG()
osg_tvmCpu = TvmCpuOsg()
osg_vhdl4cnn = vhdl4cnnOSG()
# osg_tips = TipsOSG()
# osg_vta = VtaOSG()


# fpga_OSGs = [osg_hls4ml, osg_vhdl4cnn, osg_tips, osg_vta]
fpga_OSGs = [osg_hls4ml, osg_vhdl4cnn]
# builtin_OSGs = [osg_hls4ml, osg_vhdl4cnn, osg_tvmCpu]
builtin_OSGs = [osg_hls4ml, osg_vhdl4cnn]


def merge_ops_dict(osgs):
    """merge all callable entries"""
    merged_relay2ops = {}

    # add all remaining options
    for e in relay_op_list:
        if e not in merged_relay2ops:
            merged_relay2ops[e] = relay_op_list[e]
    return merged_relay2ops

