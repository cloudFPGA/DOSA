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
import json

from dimidium.backend.devices.dosa_device import DosaHwClasses
from dimidium.lib.units import kiloU
from dimidium.lib.util import BrickImplTypes
from dimidium.middleend.archGen.DosaContract import DosaContract


def sort_brick_contracts_by_util(contr_list, consider_switching=False):
    if not consider_switching:
        nl = sorted(contr_list, key=lambda c: max(c.comp_util_share, c.mem_util_share))
    else:
        nl = sorted(contr_list, key=lambda c: max(c.comp_util_share + c.switching_comp_share,
                                                  c.mem_util_share + c.switching_mem_share))
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
        # self.osg_intern_id = ''
        self.detailed_FPGA_component_share = {}
        self.detailed_FPGA_component_share['LUTLOG']    = 0.0
        self.detailed_FPGA_component_share['LUTMEM']    = 0.0
        self.detailed_FPGA_component_share['Registers'] = 0.0
        self.detailed_FPGA_component_share['BRAM']      = 0.0
        self.detailed_FPGA_component_share['DSPs']      = 0.0
        self.detailed_FPGA_wrapper_share = {}
        self.detailed_FPGA_wrapper_share['LUTLOG']    = 0.0
        self.detailed_FPGA_wrapper_share['LUTMEM']    = 0.0
        self.detailed_FPGA_wrapper_share['Registers'] = 0.0
        self.detailed_FPGA_wrapper_share['BRAM']      = 0.0
        self.detailed_FPGA_wrapper_share['DSPs']      = 0.0
        self.is_pseudo_contract = False
        self._combine_op_contracts()

    def _combine_op_contracts(self):
        self.comp_util_share = 0
        self.mem_util_share = 0
        self.switching_comp_share = -1
        self.switching_mem_share = -1
        self.total_bytes = 0
        first_total_bytes = None
        self.oi_iter = 0
        self.detailed_FPGA_component_share['LUTLOG']    = 0.0
        self.detailed_FPGA_component_share['LUTMEM']    = 0.0
        self.detailed_FPGA_component_share['Registers'] = 0.0
        self.detailed_FPGA_component_share['BRAM']      = 0.0
        self.detailed_FPGA_component_share['DSPs']      = 0.0
        self.detailed_FPGA_wrapper_share = None
        for opc in self.op_contracts:
            if self.impl_type != opc.impl_type or self.device != opc.device or self.osg != opc.osg:
                print("[DOSA:contracts:ERROR] Trying to combine un-compatible contracts. STOP.")
                exit(1)
            if opc.iter_hz < self.iter_hz:
                self.iter_hz = opc.iter_hz
            # if self.osg_intern_id == '':
            #     self.osg_intern_id = opc.osg_intern_id
            # elif self.osg_intern_id != opc.osg_intern_id:
            #     self.osg_intern_id = 'multiple'
            if opc.switching_comp_share > self.switching_comp_share \
                    or opc.switching_mem_share > self.switching_mem_share:
                self.switching_comp_share = opc.switching_comp_share
                self.switching_mem_share = opc.switching_mem_share
                self.detailed_FPGA_wrapper_share = opc.detailed_FPGA_wrapper_share
            if self.impl_type == BrickImplTypes.ENGINE:
                if opc.comp_util_share > self.comp_util_share:
                    self.comp_util_share = opc.comp_util_share
                if opc.mem_util_share > self.mem_util_share:
                    self.mem_util_share = opc.mem_util_share
            else:
                self.comp_util_share += opc.comp_util_share
                self.mem_util_share += opc.mem_util_share
            # self.total_bytes += opc.total_bytes  # all are engine or stream -> so is correct
            # -> NO...would calc internal output/input
            if first_total_bytes is None:
                first_total_bytes = opc.total_bytes
                self.total_bytes += opc.op.input_bytes
            if self.impl_type == BrickImplTypes.ENGINE:
                self.total_bytes += opc.op.parameter_bytes
            if opc.detailed_FPGA_component_share is not None:
                if self.impl_type != BrickImplTypes.ENGINE:
                    self.detailed_FPGA_component_share['LUTLOG']          += opc.detailed_FPGA_component_share['LUTLOG']
                    self.detailed_FPGA_component_share['LUTMEM']          += opc.detailed_FPGA_component_share['LUTMEM']
                    self.detailed_FPGA_component_share['Registers']       += opc.detailed_FPGA_component_share['Registers']
                    self.detailed_FPGA_component_share['BRAM']            += opc.detailed_FPGA_component_share['BRAM']
                    self.detailed_FPGA_component_share['DSPs']            += opc.detailed_FPGA_component_share['DSPs']
                else:
                    for k in self.detailed_FPGA_component_share:
                        if opc.detailed_FPGA_component_share[k] > self.detailed_FPGA_component_share[k]:
                            self.detailed_FPGA_component_share[k] = opc.detailed_FPGA_component_share[k]
        # self.oi_iter = self.iter_hz / self.total_bytes
        # self.oi_iter = kiloU / self.total_bytes
        # self.oi_iter = self.total_bytes
        # self.oi_iter = self.iter_hz / first_total_bytes
        if self.total_bytes > 0:
            self.oi_iter = 1 / self.total_bytes
        # assert self.iter_hz < float('inf')

    def __repr__(self):
        return "BrickContr({} on {} using {}/{}: {:.2f}/s, {:.2f}c%, {:.2f}m%, switching {:.2f}%c, {:.2f}%m)" \
            .format(self.brick.fn_label, self.device.name, self.osg.name, self.impl_type, float(self.iter_hz),
                    self.comp_util_share*100, self.mem_util_share*100, self.switching_comp_share*100,
                    self.switching_mem_share*100)

    def as_dict(self):
        res = {'osg': str(self.osg.name), 'impl_type': str(self.impl_type), 'iter_hz': float(self.iter_hz),
               'device': self.device.name,
               'comp_share_%:': self.comp_util_share*100, 'mem_share_%': self.mem_util_share*100,
               'switching_comp_share_%': self.switching_comp_share*100,
               'switching_mem_share_%': self.switching_mem_share*100,
               'oi_iter': float(self.oi_iter),
               'component_utility_detail': {},
               'wrapper_utility_detail': {}}
        fpga_utility = self.device.get_resource_dict()['FPGA_utility']
        comp_util_detail = {}
        comp_util_detail['LUTLOG']     = self.detailed_FPGA_component_share['LUTLOG']    * fpga_utility['LUTLOG']
        comp_util_detail['LUTMEM']     = self.detailed_FPGA_component_share['LUTMEM']    * fpga_utility['LUTMEM']
        comp_util_detail['Registers']  = self.detailed_FPGA_component_share['Registers'] * fpga_utility['Registers']
        comp_util_detail['BRAM']       = self.detailed_FPGA_component_share['BRAM']      * fpga_utility['BRAM']
        comp_util_detail['DSPs']       = self.detailed_FPGA_component_share['DSPs']      * fpga_utility['DSPs']
        res['component_utility_detail'] = comp_util_detail
        wrapper_util_detail = {}
        if self.detailed_FPGA_wrapper_share is not None:
            wrapper_util_detail['LUTLOG']     = self.detailed_FPGA_wrapper_share['LUTLOG']    * fpga_utility['LUTLOG']
            wrapper_util_detail['LUTMEM']     = self.detailed_FPGA_wrapper_share['LUTMEM']    * fpga_utility['LUTMEM']
            wrapper_util_detail['Registers']  = self.detailed_FPGA_wrapper_share['Registers'] * fpga_utility['Registers']
            wrapper_util_detail['BRAM']       = self.detailed_FPGA_wrapper_share['BRAM']      * fpga_utility['BRAM']
            wrapper_util_detail['DSPs']       = self.detailed_FPGA_wrapper_share['DSPs']      * fpga_utility['DSPs']
        else:
            wrapper_util_detail['INFO'] = "no wrapper present"
        res['wrapper_utility_detail'] = wrapper_util_detail
        return res

    def __str__(self):
        ret = self.as_dict()
        return json.dumps(ret, indent=2)

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

    def add_op_contract(self, opc):
        self.op_contracts.append(opc)
        self._combine_op_contracts()

    def del_op_contract(self, opc):
        try:
            del_i = self.op_contracts.index(opc)
            del self.op_contracts[del_i]
            self._combine_op_contracts()
        except:
            return


