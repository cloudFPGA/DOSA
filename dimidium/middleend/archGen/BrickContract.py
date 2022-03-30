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
from dimidium.backend.devices.dosa_device import DosaHwClasses
from dimidium.middleend.archGen.DosaContract import DosaContract


def sort_brick_contracts_by_util(contr_list):
    nl = sorted(contr_list, key=lambda c: max(c.comp_util_share, c.mem_util_share))
    return nl


def sort_brick_contracts_by_iter(contr_list):
    nl = sorted(contr_list, key=lambda c: c.iter_hz, reverse=True)
    return nl


def filter_brick_contracts_by_impl_type(contr_list, selected_impl_type):
    nl = []
    for c in contr_list:
        if c.impl_type == selected_impl_type:
            nl.append(c)
    return nl


def get_best_contract_of_list(contr_list, filter_impl_type=None, filter_osg=None, filter_device=None,
                              consider_util=False, skip_entries=0):
    cur_entry_found = 0
    for c in contr_list:
        if filter_impl_type is not None and c.impl_type != filter_impl_type:
            continue
        if filter_osg is not None and c.osg != filter_osg:
            continue
        if filter_device is not None and c.device != filter_device:
            continue
        if consider_util:
            if c.comp_util_share > 1.0:
                continue
            if c.mem_util_share > 1.0:
                continue
        # else...hit
        if cur_entry_found >= skip_entries:
            return c
        else:
            cur_entry_found += 1
    return None


class BrickContract(DosaContract):

    # def __init__(self, brick: ArchBrick, device: DosaBaseHw, osg: BaseOSG, impl_type: BrickImplTypes,
    #              op_contracts: [OperationContract]):
    def __init__(self, brick, device, osg, impl_type, op_contracts):
        super().__init__(device, osg, impl_type, float('inf'), 0, 0)
        # self.iter_hz = float('inf')
        self.brick = brick
        # self.device = device
        # self.osg = osg
        # self.impl_type = impl_type
        self.op_contracts = op_contracts
        self.flops_per_iter = brick.flops
        self.comp_util_share = 0
        self.mem_util_share = 0
        self.switching_comp_share = -1
        self.switching_mem_share = -1
        self.total_bytes = 0
        self.oi_iter = 0
        self.detailed_FPGA_component_share = {}
        self.detailed_FPGA_component_share['LUTLOG']    = 0.0
        self.detailed_FPGA_component_share['LUTMEM']    = 0.0
        self.detailed_FPGA_component_share['Registers'] = 0.0
        self.detailed_FPGA_component_share['BRAM']      = 0.0
        self.detailed_FPGA_component_share['DSPs']      = 0.0
        self.detailed_FPGA_wrapper_share = None
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
                self.detailed_FPGA_wrapper_share = opc.detailed_FPGA_wrapper_share
            self.comp_util_share += opc.comp_util_share
            self.mem_util_share += opc.mem_util_share
            self.total_bytes += opc.total_bytes  # all are engine or stream -> so is correct
            if opc.detailed_FPGA_component_share is not None:
                self.detailed_FPGA_component_share['LUTLOG']          += opc.detailed_FPGA_component_share['LUTLOG']
                self.detailed_FPGA_component_share['LUTMEM']          += opc.detailed_FPGA_component_share['LUTMEM']
                self.detailed_FPGA_component_share['Registers']       += opc.detailed_FPGA_component_share['Registers']
                self.detailed_FPGA_component_share['BRAM']            += opc.detailed_FPGA_component_share['BRAM']
                self.detailed_FPGA_component_share['DSPs']            += opc.detailed_FPGA_component_share['DSPs']
        self.oi_iter = self.total_bytes / self.iter_hz

    def __repr__(self):
        return "BrickContr({} on {} using {}/{}: {}/s, {}c%, {}m%, switching {}%c, {}%m)" \
            .format(self.brick.fn_label, self.device.name, self.osg.name, self.impl_type, self.iter_hz,
                    self.comp_util_share, self.mem_util_share, self.switching_comp_share, self.switching_mem_share)

    def get_contract_to_op(self, op):
        for opc in self.op_contracts:
            if opc.op == op:
                return opc
        return None

    def ensure_detailed_utility_fits(self, consider_wrapper=False) -> bool:
        if self.device.hw_class in [DosaHwClasses.UNDECIDED, DosaHwClasses.CPU_generic, DosaHwClasses.CPU_x86]:
            return True
        if self.detailed_FPGA_component_share is None or self.detailed_FPGA_wrapper_share is None:
            return True
        for utk in self.detailed_FPGA_component_share:
            if consider_wrapper:
                if self.detailed_FPGA_component_share[utk] + self.detailed_FPGA_wrapper_share[utk] > 1.0:
                    return False
            else:
                if self.detailed_FPGA_component_share[utk] > 1.0:
                    return False
        return True


