#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: July 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Class of the architectural operation that are part of DOSA bricks
#  *
#  *

import json
from tvm.relay import Expr


class ArchOp(object):
    """
    dpl = {'name': my_name, 'cmpl': oi_cmpl, 'uinp': oi_uinp, 'flop': flop_total, 'parB': bw_param_B,
           'inpB': bw_data_B, 'outB': out_bw, 'layer': istr, 'fn': obj.cur_fstr, 'op': op_name,
           'dtype': used_dtype}
    """

    def __init__(self, op_id=None, dpl_dict=None, tvm_node=None):
        self.name = None
        self.op_id = op_id
        self.oi_engine = 0
        self.oi_stream = 0
        self.flops = 0
        self.parameter_bytes = 0
        self.input_bytes = 0
        self.output_bytes = 0
        self.layer_name = 0
        self.parent_fn = None
        self.op_call = None
        self.used_dtype = None
        self.tvm_node = tvm_node
        if dpl_dict is not None:
            self.from_dpl_dict(dpl_dict)

    def __repr__(self):
        return "ArchOp({})".format(self.op_call)

    def as_dict(self):
        res = {'name': self.name, 'oi_engine': self.oi_engine, 'oi_stream': self.oi_stream, 'flops': self.flops,
               'parameter_bytes': self.parameter_bytes, 'input_bytes': self.input_bytes,
               'output_bytes': self.output_bytes, 'layer_name': self.layer_name, 'parent_fn': self.parent_fn,
               'op_call': self.op_call, 'used_dtype': self.used_dtype, 'tvm_node': str(self.tvm_node)[:100]}
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
        self.layer_name = dpl_dict['layer']
        self.parent_fn = dpl_dict['fn']
        self.op_call = dpl_dict['op']
        self.used_dtype = dpl_dict['dtype']

    def set_op_id(self, op_id):
        self.op_id = op_id

    def set_tvm_node(self, tvm_node: Expr):
        self.tvm_node = tvm_node

