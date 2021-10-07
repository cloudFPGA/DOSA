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
from tvm.relay import Expr

from dimidium.middleend.archGen.ArchOp import ArchOp
from dimidium.lib.util import BrickImplTypes


class ArchBrick(object):

    """
    dpl = {'name': my_name, 'cmpl': oi_cmpl, 'uinp': oi_uinp, 'flop': flop_total, 'parB': bw_param_B,
           'inpB': bw_data_B, 'outB': out_bw, 'layer': istr, 'fn': obj.cur_fstr, 'op': op_name,
           'dtype': used_dtype}
    """

    def __init__(self, brick_id=None, dpl_dict=None, tvm_node=None):
        self.name = None
        self.local_brick_id = brick_id
        self.brick_uuid = None
        self.oi_engine = 0
        self.oi_stream = 0
        self.flops = 0
        self.parameter_bytes = 0
        self.input_bytes = 0
        self.output_bytes = 0
        self.fn_label = 0
        # self.parent_fn = None
        # self.op_call = None
        self.used_dtype = None
        self.tvm_node = tvm_node
        self.ops = {}
        self.oid_cnt = 0
        if dpl_dict is not None:
            self.from_dpl_dict(dpl_dict)
        self.req_flops = -1
        self.req_flops_engine = -1
        self.req_flops_stream = -1
        self.input_bw_Bs = -1
        self.output_bw_Bs = -1
        self.calc_latency = -1
        self.req_latency = -1
        self.selected_impl_type = BrickImplTypes.UNDECIDED
        self.calc_flops = -1

    def __repr__(self):
        return "ArchBrick({}, {})".format(self.local_brick_id, self.name)

    def as_dict(self):
        res = {'name': self.name, 'brick_uuid': self.brick_uuid,
               'oi_engine': self.oi_engine, 'oi_stream': self.oi_stream, 'flops': self.flops,
               'parameter_bytes': self.parameter_bytes, 'input_bytes': self.input_bytes,
               'output_bytes': self.output_bytes, 'fn_label': self.fn_label, 'used_dtype': self.used_dtype,
               'tvm_node': str(self.tvm_node)[:100], 'ops': {}, 'req_perf': self.req_flops,
               'input_Bs': self.input_bw_Bs, 'output_Bs': self.output_bw_Bs}
        for oi in self.ops:
            o = self.ops[oi]
            res['ops'][oi] = o.as_dict()
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
        self.used_dtype = dpl_dict['dtype']

    def set_brick_id(self, brick_id):
        self.local_brick_id = brick_id

    def set_tvm_node(self, tvm_node: Expr):
        self.tvm_node = tvm_node

    def add_arch_op(self, op: ArchOp):
        o_id = self.oid_cnt
        self.oid_cnt += 1
        op.set_local_op_id(o_id)
        self.ops[o_id] = op

    def set_impl_type(self, it: BrickImplTypes):
        self.selected_impl_type = it
        if it == BrickImplTypes.STREAM:
            self.req_flops_stream = self.req_flops
            self.req_flops_engine = -1
        elif it == BrickImplTypes.ENGINE:
            self.req_flops_engine = self.req_flops
            self.req_flops_stream = -1

    def update_global_ids(self, gid_start):
        next_gid = gid_start
        for op in self.local_op_iter_gen():
            op.set_global_op_id(next_gid)
            next_gid += 1
        return next_gid

    def set_brick_uuid(self, buuid):
        self.brick_uuid = buuid
