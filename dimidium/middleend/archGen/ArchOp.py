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
from types import SimpleNamespace
from tvm.relay import Expr

import dimidium.lib.singleton as dosa_singleton
from dimidium.backend.operatorSets.BaseOSG import placeholderOSG, BaseOSG
from dimidium.lib.dosa_dtype import DosaDtype, convert_tvmDtype_to_DosaDtype
from dimidium.lib.dtype_converters import get_flops_conv_factor


class ArchOp(object):
    """
    dpl = {'name': my_name, 'cmpl': oi_cmpl, 'uinp': oi_uinp, 'flop': flop_total, 'parB': bw_param_B,
           'inpB': bw_data_B, 'outB': out_bw, 'layer': istr, 'fn': obj.cur_fstr, 'op': op_name,
           'dtype': used_dtype}
    """

    def __init__(self, op_id=None, dpl_dict=None, tvm_node=None, tvm_args=None):
        self.name = None
        self.local_op_id = op_id
        self.global_op_id = None
        self.kernel_uuid = self.global_op_id  # alias, to be sure
        self.oi_engine = 0
        self.oi_stream = 0
        self.flops = 0
        self.parameter_bytes = 0
        self.input_bytes = 0
        self.output_bytes = 0
        self.layer_name = 0
        self.dims = SimpleNamespace()
        self.dims.inp = 0
        self.dims.out = 0
        self.dims.param = 0
        self.parent_fn = None
        self.op_call = None
        self.used_dtype = DosaDtype.UNKNOWN
        self.flops_conv_factor = dosa_singleton.config.dtype.default_dosa_flops_conv_factor
        self.tvm_dtype = None
        self.tvm_node = tvm_node
        self.tvm_args = tvm_args
        self.selected_osg = placeholderOSG
        self.possible_osgs = []
        if dpl_dict is not None:
            self.from_dpl_dict(dpl_dict)

    def __repr__(self):
        return "ArchOp({})".format(self.op_call)

    def as_dict(self):
        res = {'name': self.name, 'local_id': self.local_op_id, 'global_id': self.global_op_id,
               'oi_engine': self.oi_engine, 'oi_stream': self.oi_stream, 'flops': self.flops,
               'parameter_bytes': self.parameter_bytes, 'input_bytes': self.input_bytes,
               'output_bytes': self.output_bytes, 'layer_name': self.layer_name, 'parent_fn': self.parent_fn,
               'op_call': self.op_call, 'used_dtype': repr(self.used_dtype), 'tvm_node': str(self.tvm_node)[:100]} #,
               # 'possible OSGs': [], 'selected OSG': repr(self.selected_osg)}
        # for po in self.possible_osgs:
        #     pos = repr(po)
        #     res['possible OSGs'].append(pos)
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
        self.tvm_dtype = dpl_dict['dtype']
        self.used_dtype = convert_tvmDtype_to_DosaDtype(self.tvm_dtype)
        self.flops_conv_factor = get_flops_conv_factor(self.used_dtype)
        self.dims.inp = dpl_dict['dims']['inp']
        if len(self.dims.inp) > 0 and type(self.dims.inp[0]) is list:
            self.dims.inp = self.dims.inp[0]
            # the other entry is likely in params
        self.dims.param = dpl_dict['dims']['param']
        if len(self.dims.param) > 0 and type(self.dims.param[0]) is list:
            self.dims.param = self.dims.param[0]
        self.dims.out = dpl_dict['dims']['out']
        if len(self.dims.out) > 0 and type(self.dims.out[0]) is list:
            self.dims.out = self.dims.out[0]

    def set_local_op_id(self, op_id):
        self.local_op_id = op_id

    def set_global_op_id(self, gid):
        self.global_op_id = gid

    def set_tvm_node(self, tvm_node: Expr):
        self.tvm_node = tvm_node

    def set_tvm_args(self, tvm_arg_dict):
        self.tvm_args = tvm_arg_dict

    def get_kernel_uuid(self):
        return self.global_op_id

    def set_osg(self, osg: BaseOSG):
        self.selected_osg = osg

    def add_possible_osg(self, osg: BaseOSG):
        self.possible_osgs.append(osg)
        self.possible_osgs = list(set(self.possible_osgs))

    def remove_possible_osg(self, osg: BaseOSG):
        delme = self.possible_osgs.index(osg)
        del self.possible_osgs[delme]
