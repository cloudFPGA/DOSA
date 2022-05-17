#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: July 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Class of the architectural bricks for DOSA
#  *
#  *

import json
from types import SimpleNamespace

from tvm.relay import Expr

import dimidium.lib.singleton as dosa_singleton
from dimidium.backend.devices.dosa_device import DosaBaseHw
from dimidium.middleend.archGen.ArchOp import ArchOp
from dimidium.lib.util import BrickImplTypes
from dimidium.lib.dosa_dtype import DosaDtype, convert_tvmDtype_to_DosaDtype
from dimidium.lib.dtype_converters import get_flops_conv_factor
from dimidium.backend.operatorSets.BaseOSG import placeholderOSG, BaseOSG, sort_osg_list
from dimidium.middleend.archGen.BrickContract import BrickContract, filter_brick_contracts_by_impl_type, \
    sort_brick_contracts_by_iter, sort_brick_contracts_by_util, get_best_contract_of_list
from dimidium.middleend.archGen.DosaContract import DosaContract
from dimidium.middleend.archGen.parallelizeBrick import parallelize_ops_of_brick


class ArchBrick(object):
    """
    dpl = {'name': my_name, 'cmpl': oi_cmpl, 'uinp': oi_uinp, 'flop': flop_total, 'parB': bw_param_B,
           'inpB': bw_data_B, 'outB': out_bw, 'layer': istr, 'fn': obj.cur_fstr, 'op': op_name,
           'dtype': used_dtype}
    """

    def __init__(self, brick_id=None, dpl_dict=None, tvm_node=None, tvm_args=None):
        self.name = None
        self.local_brick_id = brick_id
        self.brick_uuid = None
        self.oi_engine = 0
        self.oi_stream = 0
        self.flops = 0
        self.iter_hz = 0
        self.oi_iter = 0
        self.used_flops = -1
        self.parameter_bytes = 0
        self.input_bytes = 0
        self.output_bytes = 0
        self.fn_label = ''
        # self.parent_fn = None
        # self.op_call = None
        self.used_dtype = DosaDtype.UNKNOWN
        self.flops_conv_factor = dosa_singleton.config.dtype.default_dosa_flops_conv_factor
        self.tvm_dtype = None
        self.tvm_node = tvm_node
        self.tvm_args = tvm_args
        self.ops = {}
        self.oid_cnt = 0
        if dpl_dict is not None:
            self.from_dpl_dict(dpl_dict)
        self.req_flops = -1
        self.req_exec_time_s = 0
        self.req_flops_engine = -1
        self.req_flops_stream = -1
        self.req_iter_hz = -1
        self.input_bw_Bs = -1
        self.output_bw_Bs = -1
        self.calc_latency = -1
        self.req_latency = -1
        self.selected_impl_type = BrickImplTypes.UNDECIDED
        self.calc_flops = -1
        self.selected_osg = placeholderOSG
        self.possible_osgs = []
        self.available_osgs = []
        self.available_contracts = []
        self.still_possible_contracts = []
        self.selected_contract = None
        self.max_possible_iter = -1
        self.possible_hw_types = []
        self.req_util_comp = 0
        self.req_util_comp_engine = 0
        self.req_util_comp_stream = 0
        self.req_util_mem = 0
        self.req_util_mem_engine = 0
        self.req_util_mem_stream = 0
        self.switching_comp_share = 0
        self.switching_mem_share = 0
        self.tmp_osg = None
        self.skip_in_roofline = False
        self.dims = SimpleNamespace()
        self.dims.inp = 0
        self.dims.out = 0
        self.dims.param = 0
        self.local_pipeline_store = 0
        self.needs_compute_parallelization = False
        self.parallelized_bricks = None
        self.compute_parallelization_factor = 1

    def __repr__(self):
        return "ArchBrick({}, {})".format(self.local_brick_id, self.name)

    def as_summary(self):
        res = {'name': self.name, 'brick_uuid': self.brick_uuid,
               'op_calls': []}
        for oi in self.ops:
            o = self.ops[oi]
            res['op_calls'].append(o.op_call)
        return res

    def as_dict(self):
        res = {'name': self.name, 'brick_uuid': self.brick_uuid,
               'oi_engine': self.oi_engine, 'oi_stream': self.oi_stream, 'flops': self.flops,
               'parameter_bytes': self.parameter_bytes, 'input_bytes': self.input_bytes,
               'output_bytes': self.output_bytes, 'fn_label': self.fn_label, 'used_dtype': repr(self.used_dtype),
               'dims': '', 'tvm_node': str(self.tvm_node)[:100], 'req_flops': self.req_flops,
               'req_latency_s': self.req_latency, 'req_iter_hz': self.req_iter_hz, 'iter_hz': self.iter_hz,
               'oi_iter': self.oi_iter, 'flops_based_on_iters': self.used_flops,
               'req_util_comp': self.req_util_comp, 'req_util_mem': self.req_util_mem,
               'input_Bs': self.input_bw_Bs, 'output_Bs': self.output_bw_Bs,
               # 'possible OSGs': [], 'selected OSG': repr(self.selected_osg),
               'possible contr': [], 'selected contr': '',
               'selected impl. type:': repr(self.selected_impl_type),
               'ops': {}}
        for oi in self.ops:
            o = self.ops[oi]
            res['ops'][oi] = o.as_dict()
        # for po in self.possible_osgs:
        #     pos = repr(po)
        #     res['possible OSGs'].append(pos)
        for po in self.available_contracts:
            pos = repr(po)
            res['possible contr'].append(pos)
        if self.selected_contract is None:
            res['selected contr'] = 'None'
        else:
            res['selected contr'] = self.selected_contract.as_dict()
        self.update_dims()
        res['dims'] = '(inp: {}, out: {}, params: {})'.format(self.dims.inp, self.dims.out, self.dims.param)
        return res

    def __str__(self):
        ret = self.as_dict()
        return json.dumps(ret, indent=2)

    def local_op_iter_gen(self):
        for oi in self.ops:
            o = self.ops[oi]
            yield o

    def from_dpl_dict(self, dpl_dict):
        self.name = dpl_dict['name']
        self.oi_engine = dpl_dict['cmpl']
        self.oi_stream = dpl_dict['uinp']
        self.flops = dpl_dict['flop']
        self.parameter_bytes = dpl_dict['parB']
        self.input_bytes = dpl_dict['inpB']
        self.output_bytes = dpl_dict['outB']
        self.fn_label = dpl_dict['layer']
        # self.parent_fn = dpl_dict['fn']
        # self.op_call = dpl_dict['op']
        self.tvm_dtype = dpl_dict['dtype']
        self.used_dtype = convert_tvmDtype_to_DosaDtype(self.tvm_dtype)
        self.flops_conv_factor = get_flops_conv_factor(self.used_dtype)
        self.update_dims()

    def reconstruct_from_op_list(self, op_list):
        self.oid_cnt = 0
        self.ops = {}
        total_flops = 0
        total_uinp = 0
        total_params = 0
        self.input_bytes = 0
        for op in op_list:
            if self.input_bytes == 0:
                # first op
                self.input_bytes = op.input_bytes
            total_flops += op.flops
            total_uinp += op.input_bytes
            total_params += op.parameter_bytes
            self.output_bytes = op.output_bytes
            self.add_arch_op(op)
        self.oi_engine = (total_uinp + total_params) / total_flops
        self.oi_stream = total_uinp / total_flops
        self.flops = total_flops
        self.parameter_bytes = total_params
        self.update_dims()

    def set_brick_id(self, brick_id):
        self.local_brick_id = brick_id

    def set_tvm_node(self, tvm_node: Expr):
        self.tvm_node = tvm_node

    def set_tvm_args(self, tvm_arg_dict):
        self.tvm_args = tvm_arg_dict

    def add_arch_op(self, op: ArchOp):
        o_id = self.oid_cnt
        self.oid_cnt += 1
        op.set_local_op_id(o_id)
        self.ops[o_id] = op
        already_considered_contr = []
        for opc in op.possible_contracts:
            for my_contr in self.available_contracts:
                # if my_contr.osg_intern_id == opc.osg_intern_id and \
                #         my_contr.osg == opc.osg and my_contr.impl_type == opc.impl_type and \
                #         my_contr.device == opc.device:
                if my_contr.osg == opc.osg and my_contr.impl_type == opc.impl_type and \
                        my_contr.device == opc.device and \
                        my_contr not in already_considered_contr:
                    my_contr.add_op_contract(opc)
                    already_considered_contr.append(my_contr)

    def del_arch_op(self, op_i):
        op = self.ops[op_i]
        for bc in self.available_contracts:
            for opc in op.possible_contracts:
                bc.del_op_contract(opc)
        del self.ops[op_i]
        for i in range(op_i + 1, self.oid_cnt):
            self.ops[i].local_op_id -= 1
        self.oid_cnt -= 1

    def annotate_parallelization(self, factor):
        self.req_flops /= factor
        self.req_util_comp /= factor
        # memory not affected
        if self.selected_impl_type == BrickImplTypes.STREAM:
            self.req_flops_stream /= factor
            self.req_util_comp_stream /= factor
        elif self.selected_impl_type == BrickImplTypes.ENGINE:
            self.req_flops_engine /= factor
            self.req_util_comp_engine /= factor

    def split(self, op_id_to_new_brick):
        if op_id_to_new_brick == 0 or op_id_to_new_brick >= self.oid_cnt or op_id_to_new_brick < 0:
            print("[DOSA:ArchNode:ERROR] invalid split attempt, skipping.")
            return None
        new_brick = ArchBrick()
        orig_name = self.name
        orig_label = self.fn_label
        self.name += '_split_0'
        self.fn_label += '_split_0'
        new_brick.name = orig_name + '_split_1'
        new_brick.fn_label = orig_label + '_split_1'
        new_brick.tvm_dtype = self.tvm_dtype
        new_brick.used_dtype = self.used_dtype
        new_brick.flops_conv_factor = self.flops_conv_factor
        new_brick.available_osgs = self.available_osgs
        new_brick.possible_osgs = self.possible_osgs
        new_brick.possible_hw_types = self.possible_hw_types
        # update required performance afterwards...skip it for now
        op_list_staying = []
        op_list_new = []
        for i in range(0, self.oid_cnt):
            op = self.ops[i]
            if i >= op_id_to_new_brick:
                op_list_new.append(op)
            else:
                op_list_staying.append(op)
        self.reconstruct_from_op_list(op_list_staying)
        new_brick.reconstruct_from_op_list(op_list_new)

    def parallelize(self, contracts_to_consider, factor, with_inputs=False):
        # self.still_possible_contracts = []
        used_factor, new_ops_dict = parallelize_ops_of_brick(self, factor * self.compute_parallelization_factor,
                                                             with_inputs=with_inputs)
        if used_factor < 0:
            print("[DOSA:ArchBrick:ERROR] Brick {} is forced to parallelize but can't. STOP.".format(self.brick_uuid))
            exit(1)
        self.compute_parallelization_factor = used_factor  # to progress on recursion
        new_brick_list = []
        for i in range(0, used_factor):
            new_brick = ArchBrick()
            new_brick.name = self.name + '_split_{}of{}'.format(i + 1, used_factor)
            new_brick.fn_label = self.fn_label + '_split_{}of{}'.format(i + 1, used_factor)
            new_brick.tvm_dtype = self.tvm_dtype
            new_brick.used_dtype = self.used_dtype
            new_brick.flops_conv_factor = self.flops_conv_factor
            new_brick.available_osgs = self.available_osgs
            new_brick.possible_osgs = self.possible_osgs
            new_brick.possible_hw_types = self.possible_hw_types
            op_list_new_brick = []
            for oid in self.ops:
                op_list_new_brick.append(new_ops_dict[oid][i])
            new_brick.reconstruct_from_op_list(op_list_new_brick)
            new_brick.orig_brick_object = self
            new_brick.selected_impl_type = self.selected_impl_type
            new_brick.available_contracts = []
            new_brick.input_bw_Bs = self.input_bw_Bs  # stays the same
            if with_inputs:
                new_brick.input_bw_Bs = self.input_bw_Bs / factor
            new_brick.output_bw_Bs = self.output_bw_Bs / factor
            new_brick.req_iter_hz = self.req_iter_hz
            new_brick.req_latency = self.req_latency
            new_brick_list.append(new_brick)
        self.parallelized_bricks = new_brick_list
        considered_osgs = []
        considered_devices = []
        for cc in contracts_to_consider:
            if cc.osg not in considered_osgs and cc.device not in considered_devices:
                considered_osgs.append(cc.osg)
                considered_devices.append(cc.device)
                # later
                # for nb in self.parallelized_bricks:
                #     cc.osg.annotate_brick(nb, cc.device)
                # add fake contract
                pseudo_contract = BrickContract(self, cc.device, cc.osg, cc.impl_type, [])
                pseudo_contract.iter_hz = cc.iter_hz / used_factor
                pseudo_contract.flops_per_iter = cc.flops_per_iter / used_factor
                pseudo_contract.comp_util_share = cc.comp_util_share / used_factor
                pseudo_contract.mem_util_share = cc.mem_util_share / used_factor
                pseudo_contract.switching_comp_share = cc.switching_comp_share  # switching cost's don't change?
                pseudo_contract.switching_mem_share = cc.switching_mem_share
                pseudo_contract.total_bytes = cc.total_bytes / used_factor
                pseudo_contract.oi_iter = 1 / pseudo_contract.total_bytes
                pseudo_contract.op_contracts = [None] * self.oid_cnt
                pseudo_contract.is_pseudo_contract = True
                self.add_possible_contract(pseudo_contract)
        self.needs_compute_parallelization = True

    def set_impl_type(self, it: BrickImplTypes):
        self.selected_impl_type = it
        if it == BrickImplTypes.STREAM:
            self.req_flops_stream = self.req_flops
            self.req_flops_engine = -1
            self.req_util_mem = self.req_util_mem_stream
            self.req_util_comp = self.req_util_comp_stream
        elif it == BrickImplTypes.ENGINE:
            self.req_flops_engine = self.req_flops
            self.req_flops_stream = -1
            self.req_util_mem = self.req_util_mem_engine
            self.req_util_comp = self.req_util_comp_engine

    def get_oi_selected_impl(self, fallback_impl_type=BrickImplTypes.ENGINE):
        if self.selected_impl_type == BrickImplTypes.STREAM:
            return self.oi_stream
        elif self.selected_impl_type == BrickImplTypes.ENGINE:
            return self.oi_engine
        else:
            if fallback_impl_type == BrickImplTypes.STREAM:
                return self.oi_stream
            return self.oi_engine

    def update_global_ids(self, gid_start):
        next_gid = gid_start
        for op in self.local_op_iter_gen():
            op.set_global_op_id(next_gid)
            next_gid += 1
        return next_gid

    def set_brick_uuid(self, buuid):
        self.brick_uuid = buuid

    # def set_osg(self, osg: BaseOSG):
    #     self.selected_osg = osg

    def set_contract(self, contr: BrickContract):
        assert self.selected_impl_type == contr.impl_type
        self.selected_contract = contr
        self.selected_osg = contr.osg
        self.iter_hz = contr.iter_hz
        self.oi_iter = contr.oi_iter
        # self.flops = self.iter_hz * contr.flops_per_iter
        self.used_flops = self.iter_hz * contr.flops_per_iter
        self.update_util_estimation_contr(contr.device)

    # def add_possible_osg(self, osg: BaseOSG):
    #     self.possible_osgs.append(osg)
    #     self.possible_osgs = list(set(self.possible_osgs))

    # def remove_possible_osg(self, osg: BaseOSG):
    #     delme = self.possible_osgs.index(osg)
    #     del self.possible_osgs[delme]

    # def add_available_osg(self, osg: BaseOSG):
    #     self.available_osgs.append(osg)
    #     self.available_osgs = list(set(self.available_osgs))

    # def update_possible_osgs(self):
    #     # find all possible osgs, based on ops
    #     cur_possible_osgs = self.available_osgs
    #     not_possible_osgs = []
    #     for op in self.local_op_iter_gen():
    #         op_posg = op.possible_osgs
    #         for bpo in cur_possible_osgs:
    #             if bpo not in op_posg:
    #                 not_possible_osgs.append(bpo)
    #     not_possible_osgs = list(set(not_possible_osgs))
    #     for npo in not_possible_osgs:
    #         del cur_possible_osgs[cur_possible_osgs.index(npo)]
    #     # remove osgs based on impl type
    #     tmp_osg_list = sort_osg_list(cur_possible_osgs)
    #     not_possible_osgs = []
    #     if self.selected_impl_type != BrickImplTypes.UNDECIDED:
    #         for posg in tmp_osg_list:
    #             if self.selected_impl_type not in posg.possible_impl_types:
    #                 not_possible_osgs.append(posg)
    #     for npo in not_possible_osgs:
    #         del tmp_osg_list[tmp_osg_list.index(npo)]
    #     self.possible_osgs = sort_osg_list(tmp_osg_list)

    def add_possible_contract(self, contr: BrickContract):
        assert contr.brick == self
        self.available_contracts.append(contr)

    def sort_contracts(self, by_utility=False):
        """sort possible and available contracts by performance (default) or by used utility"""
        if not by_utility:
            self.available_contracts = sort_brick_contracts_by_iter(self.available_contracts)
            self.still_possible_contracts = sort_brick_contracts_by_iter(self.still_possible_contracts)
        else:
            self.available_contracts = sort_brick_contracts_by_util(self.available_contracts)
            self.still_possible_contracts = sort_brick_contracts_by_util(self.still_possible_contracts)

    def get_best_available_contract(self, filter_impl_type=None, filter_osg=None, filter_device=None,
                                    consider_util=False, skip_entries=0):
        # assume sorted
        return get_best_contract_of_list(self.available_contracts, filter_impl_type, filter_osg, filter_device,
                                         consider_util, skip_entries)

    def get_best_possible_contract(self, filter_impl_type=None, filter_osg=None, filter_device=None, skip_entries=0):
        # assume sorted
        return get_best_contract_of_list(self.still_possible_contracts, filter_impl_type, filter_osg, filter_device,
                                         True, skip_entries)

    def get_best_sufficient_contract_with_least_resources(self):
        possible_by_util = sort_brick_contracts_by_util(self.still_possible_contracts)
        # selected_contract = self.get_best_possible_contract()
        if len(possible_by_util) == 0:
            return None
        if len(possible_by_util) == 1:
            # only one possible --> already decided
            return possible_by_util[0]
        selected_contract = DosaContract(None, None, None, 0, 10.0, 10.0)
        for next_poc in possible_by_util:
            if next_poc.iter_hz >= self.req_iter_hz and \
                    (next_poc.comp_util_share < selected_contract.comp_util_share and
                     next_poc.mem_util_share < selected_contract.mem_util_share) and \
                    next_poc.iter_hz >= selected_contract.iter_hz:
                selected_contract = next_poc
        if selected_contract.device is None:
            # run again with relaxed conditions, but start with best possible performance
            possible_by_util.sort(key=lambda c: c.iter_hz, reverse=True)
            for next_poc in possible_by_util:
                if (next_poc.comp_util_share < selected_contract.comp_util_share and
                    next_poc.mem_util_share < selected_contract.mem_util_share) and \
                        (next_poc.iter_hz >= selected_contract.iter_hz):
                    selected_contract = next_poc
        return selected_contract

    def update_possible_contracts(self, consider_switching=False, assume_osg=None):
        still_possible = []
        within_util_exception = []
        fitting_type = []
        for c in self.available_contracts:
            if self.selected_impl_type != BrickImplTypes.UNDECIDED and c.impl_type != self.selected_impl_type:
                continue
            if len(c.op_contracts) != len(self.ops):
                continue
            # device is set?
            # osg not relevant?
            fitting_type.append(c)
            consider_wrapper = consider_switching
            if consider_switching:
                if c.osg == assume_osg:
                    consider_wrapper = False
            if not c.ensure_detailed_utility_fits(consider_wrapper=consider_wrapper):
                continue
            total_comp_share = c.comp_util_share
            total_mem_share = c.mem_util_share
            if consider_wrapper:
                total_comp_share += c.switching_comp_share
                total_mem_share += c.switching_mem_share
            if total_comp_share > dosa_singleton.config.utilization.dosa_xi or \
                    total_mem_share > dosa_singleton.config.utilization.dosa_xi:
                if not (total_comp_share > dosa_singleton.config.utilization.dosa_xi_exception or
                        total_mem_share > dosa_singleton.config.utilization.dosa_xi_exception):
                    within_util_exception.append(c)
                # to big in all cases
                continue
            still_possible.append(c)
        if len(still_possible) == 0 and len(within_util_exception) > 0:
            print('[DOSA:ContrMngt:INFO] Brick {}: Using contract above utilization target, but within exception, '
                  'because no other contract is available.'.format(self.brick_uuid))
            self.still_possible_contracts = within_util_exception
        elif len(still_possible) == 0 and len(within_util_exception) == 0 and len(fitting_type) > 0:
            print(
                '[DOSA:ContrMngt:INFO] Brick {}: Need to parallelize, due to no available contract withing utilization '
                'bounds.'.format(self.brick_uuid))
            # if self.brick_uuid is None:
            #     print('here')
            least_split_factor = float('inf')
            for c in fitting_type:
                # cf = max(c.comp_util_share, c.mem_util_share) \
                #      / (dosa_singleton.config.utilization.dosa_xi - max(c.switching_comp_share, c.switching_mem_share))
                if consider_switching:
                    cf = round((max(c.comp_util_share, c.mem_util_share) + max(c.switching_comp_share, c.switching_mem_share)) \
                         / dosa_singleton.config.utilization.dosa_xi, 1)
                else:
                    cf = round(max(c.comp_util_share, c.mem_util_share) / dosa_singleton.config.utilization.dosa_xi, 1)
                if cf < least_split_factor:
                    least_split_factor = cf
            assert least_split_factor < float('inf')
            self.parallelize(fitting_type, least_split_factor)
            self.update_possible_contracts(consider_switching=False)
        else:
            self.still_possible_contracts = still_possible

    # def update_possible_hw_types(self):
    #     new_possible_hw_types = []
    #     for osg in self.possible_osgs:
    #         new_possible_hw_types.extend(osg.dosaHwTypes)
    #     self.possible_hw_types = list(set(new_possible_hw_types))

    def update_possible_hw_types(self):
        new_possible_hw_types = []
        for contr in self.available_contracts:
            new_possible_hw_types.extend(contr.osg.dosaHwTypes)
        self.possible_hw_types = list(set(new_possible_hw_types))

    def update_dims(self):
        self.dims = SimpleNamespace()
        self.dims.inp = None
        self.dims.out = None
        self.dims.param = []
        for lb in self.local_op_iter_gen():
            if self.dims.inp is None:
                self.dims.inp = lb.dims.inp
            self.dims.out = lb.dims.out
            self.dims.param.append(lb.dims.param)

    def update_util_estimation(self, target_hw: DosaBaseHw):
        share_comp, share_mem = target_hw.get_hw_utilization_tuple(self.req_flops, self.parameter_bytes)
        self.req_util_comp = share_comp
        self.req_util_mem = share_mem
        if self.selected_impl_type == BrickImplTypes.STREAM:
            self.req_util_mem_stream = share_mem
            self.req_util_comp_stream = share_comp
        elif self.selected_impl_type == BrickImplTypes.ENGINE:
            self.req_util_mem_engine = 0  # TODO: ? for engine, always 0!
            self.req_util_comp_engine = share_comp

    def update_util_estimation_contr(self, target_hw, prefer_engine=False):
        if self.selected_contract is None:
            self.update_possible_contracts()
            if not prefer_engine and self.selected_impl_type != BrickImplTypes.STREAM:
                tmp_best = self.get_best_possible_contract(filter_device=target_hw)
            else:
                tmp_best = self.get_best_possible_contract(filter_impl_type=BrickImplTypes.ENGINE,
                                                           filter_device=target_hw)
                if tmp_best is None:
                    tmp_best = self.get_best_possible_contract(filter_device=target_hw)
        else:
            tmp_best = self.selected_contract
        if tmp_best is None:
            print(('[DOSA:Contr:ERROR] No valid contracts left for Brick {}. Current available contracts are: {}.\n' +
                   'STOP.').format(self.brick_uuid, self.available_contracts))
            exit(1)
        share_comp = tmp_best.comp_util_share
        share_mem = tmp_best.mem_util_share
        self.switching_comp_share = tmp_best.switching_comp_share
        self.switching_mem_share = tmp_best.switching_mem_share
        self.req_util_comp = share_comp
        self.req_util_mem = share_mem
        self.iter_hz = tmp_best.iter_hz
        if self.selected_impl_type == BrickImplTypes.STREAM:
            self.req_util_mem_stream = share_mem
            self.req_util_comp_stream = share_comp
        elif self.selected_impl_type == BrickImplTypes.ENGINE:
            self.req_util_mem_engine = 0  # TODO: ? for engine, always 0!
            self.req_util_comp_engine = share_comp
        self.tmp_osg = tmp_best.osg
        max_util = max(share_comp + self.switching_comp_share, share_mem + self.switching_mem_share)
        max_iter = (1.0 / max_util) * self.iter_hz
        self.max_possible_iter = max_iter
        self.local_pipeline_store = tmp_best.osg.pipeline_tensor_store
        # self.input_bw_Bs / (self.selected_contract.device.get_performance_dict()['bw_netw_gBs'] * gigaU * 0.25 * 0.5)
