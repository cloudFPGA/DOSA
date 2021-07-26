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


class ArchBrick(object):

    """
    dpl = {'name': my_name, 'cmpl': oi_cmpl, 'uinp': oi_uinp, 'flop': flop_total, 'parB': bw_param_B,
           'inpB': bw_data_B, 'outB': out_bw, 'layer': istr, 'fn': obj.cur_fstr, 'op': op_name,
           'dtype': used_dtype}
    """

    def __init__(self, brick_id=None, dpl_dict=None):
        self.name = None
        self.brick_id = brick_id
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
        if dpl_dict is not None:
            self.from_dpl_dict(dpl_dict)

    def from_dpl_dict(self, dpl_dict):
        self.name = dpl_dict['name']
        self.oi_engine = dpl_dict['cmpl']
        self.oi_stream = dpl_dict['uinp']
        self.flops = dpl_dict['flop']
        self.parameter_bytes = dpl_dict['parB']
        self.input_bytes = dpl_dict['inpB']
        self.output_bytes = dpl_dict['outB']
        self.layer_nane = dpl_dict['layer']
        self.parent_fn = dpl_dict['fn']
        self.op_call = dpl_dict['op']
        self.used_dtype = dpl_dict['used_dtype']

    def set_brick_id(self, brick_id):
        self.brick_id = brick_id


