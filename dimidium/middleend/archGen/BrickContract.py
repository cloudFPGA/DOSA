#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Mar 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Class of implementation contracts offered by OSGs within a Brick
#  *
#  *

# from dimidium.backend.devices.dosa_device import DosaBaseHw
# from dimidium.backend.operatorSets.BaseOSG import BaseOSG
# from dimidium.lib.util import BrickImplTypes
# from dimidium.middleend.archGen.ArchBrick import ArchBrick
# from dimidium.middleend.archGen.ArchOp import ArchOp
# from dimidium.middleend.archGen.OperationContract import OperationContract


class BrickContract(object):

    # def __init__(self, brick: ArchBrick, device: DosaBaseHw, osg: BaseOSG, impl_type: BrickImplTypes,
    #              op_contracts: [OperationContract]):
    def __init__(self, brick, device, osg, impl_type, op_contracts):
        self.brick = brick
        self.device = device
        self.osg = osg
        self.impl_type = impl_type
        self.op_contracts = op_contracts
        self.iter_hz = float('inf')
        self.comp_util_share = 0
        self.mem_util_share = 0
        self.switching_comp_share = -1
        self.switching_mem_share = -1
        self._combine_op_contracts()

    def _combine_op_contracts(self):
        for opc in self.op_contracts:
            if self.impl_type != opc.impl_type or self.device != opc.device or self.osg != opc.osg:
                print("[DOSA:contracts:ERROR] Trying to combine un-compatible contracts. STOP.")
                exit(1)
            if opc.iter_hz < self.iter_hz:
                self.iter_hz = opc.iter_hz
            if opc.switching_comp_share > self.switching_comp_share \
                or opc.switching_mem_share > self.switching_mem_share:
                self.switching_comp_share = opc.switching_comp_share
                self.switching_mem_share = opc.switching_mem_share
            self.comp_util_share += opc.comp_util_share
            self.mem_util_share += opc.mem_util_share

    def __repr__(self):
        return "BrickContr({} on {} using {}/{}: {}/s, {}c%, {}m%, switching {}%c, {}%m)" \
            .format(self.brick.fn_label, self.device.name, self.osg.name, self.impl_type, self.iter_hz,
                    self.comp_util_share, self.mem_util_share, self.switching_comp_share, self.switching_mem_share)

    def get_contract_to_op(self, op):
        for opc in self.op_contracts:
            if opc.op == op:
                return opc
        return None

