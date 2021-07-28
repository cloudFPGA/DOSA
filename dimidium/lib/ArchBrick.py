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

from dimidium.lib.ArchOp import ArchOp
from dimidium.lib.util import BrickImplTypes


class ArchBrick(object):

    """
    dpl = {'name': my_name, 'cmpl': oi_cmpl, 'uinp': oi_uinp, 'flop': flop_total, 'parB': bw_param_B,
           'inpB': bw_data_B, 'outB': out_bw, 'layer': istr, 'fn': obj.cur_fstr, 'op': op_name,
           'dtype': used_dtype}
    """

    def __init__(self, brick_id=None, dpl_dict=None, tvm_node=None):
        self.name = None
        self.brick_id = brick_id
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

    def __repr__(self):
        return "ArchBrick({}, {})".format(self.brick_id, self.name)

    def as_dict(self):
        res = {'name': self.name, 'oi_engine': self.oi_engine, 'oi_stream': self.oi_stream, 'flops': self.flops,
               'parameter_bytes': self.parameter_bytes, 'input_bytes': self.input_bytes,
               'output_bytes': self.output_bytes, 'fn_labe': self.fn_label, 'used_dtype': self.used_dtype,
               'tvm_node': str(self.tvm_node)[:100], 'ops': {}, 'req_perf': self.req_flops,
               'input_Bs': self.input_bw_Bs, 'output_Bs': self.output_bw_Bs}
        for oi in self.ops:
            o = self.ops[oi]
            res['ops'][oi] = o.as_dict()
        return res

    def __str__(self):
        ret = self.as_dict()
        return json.dumps(ret, indent=2)

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
        self.brick_id = brick_id

    def set_tvm_node(self, tvm_node: Expr):
        self.tvm_node = tvm_node

    def add_arch_op(self, op: ArchOp):
        o_id = self.oid_cnt
        self.oid_cnt += 1
        op.set_op_id(o_id)
        self.ops[o_id] = op


