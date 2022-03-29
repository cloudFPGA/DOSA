#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Mar 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Class of the implementation contracts offered by OSGs
#  *
#  *

# from dimidium.backend.devices.dosa_device import DosaBaseHw
# from dimidium.backend.operatorSets.BaseOSG import BaseOSG
# from dimidium.lib.util import BrickImplTypes
# from dimidium.middleend.archGen.ArchOp import ArchOp


class OperationContract(object):

    # def __init__(self, op: ArchOp, device: DosaBaseHw, osg: BaseOSG, impl_type: BrickImplTypes,
    #              iter_hz: float, comp_util_share: float, mem_util_share: float, internal_id: str):
    def __init__(self, op, device, osg, impl_type, iter_hz: float, comp_util_share: float, mem_util_share: float,
                 internal_id: str, switching_comp_share: float, switching_mem_share: float,
                 detailed_FPGA_res=None, detailed_FPGA_wrapper=None):
        self.op = op
        self.device = device
        self.osg = osg
        self.impl_type = impl_type
        self.iter_hz = iter_hz
        self.comp_util_share = comp_util_share
        self.mem_util_share = mem_util_share
        self.osg_intern_id = internal_id
        self.switching_comp_share = switching_comp_share
        self.switching_mem_share = switching_mem_share
        self.detailed_FPGA_comp_share = detailed_FPGA_res
        self.detailed_FPGA_wrapper_share = detailed_FPGA_wrapper

    def __repr__(self):
        return "OpContr({} on {} using {}/{}: {}/s, {}c%, {}m%)"\
            .format(self.op.op_call, self.device.name, self.osg.name, self.impl_type, self.iter_hz,
                    self.comp_util_share, self.mem_util_share)






