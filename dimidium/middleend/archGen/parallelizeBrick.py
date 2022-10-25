#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Apr 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        functions to parallelize an ArchBrick
#  *
#  *
import numpy as np
import tvm.relay as relay

from dimidium.lib.dosa_dtype import get_bitwidth_of_DosaDtype
from dimidium.middleend.archGen.ArchOp import ArchOp
# from dimidium.middleend.archGen.ArchBrick import ArchBrick
from dimidium.lib.util import get_next_larger_dividor


__ops_possible_to_paralleize__ = ['nn.conv2d', 'nn.bias_add', 'tanh', 'nn.relu', 'nn.max_pool2d', 'nn.batch_flatten']
__force_to_split_inputs_after_op__ = [True,        False,      False,  False,       False,            False]
__compatible_with_splitted_input__ = [False,       True,       True,   True,         True,            True]
__min_factor__ = 2


def parallelize_ops_of_brick(orig_brick, factor_in, with_inputs=False):
    if factor_in < __min_factor__:
        factor_in = __min_factor__
    # factor = 2 * round(np.ceil(factor_in)/2)
    factor = round(np.ceil(factor_in))
    # op_force = '(round to even)'
    op_force = '(round to ceil)'
    # check factor
    necessary = True
    not_possible_factors = []
    while necessary:
        necessary = False
        for oid in orig_brick.ops:
            op = orig_brick.ops[oid]
            if with_inputs:
                if op.dims.inp[1] % factor != 0:
                    not_possible_factors.append(factor)
                    factor = get_next_larger_dividor(op.dims.inp[1], factor, not_possible_factors=not_possible_factors)
                    if factor == -1:
                        print("[DOSA:ParallelizeOpClass:ERROR] Unable to find possible split factor for op {}, dims: {}"
                              ", factors tried {}, requested {} (input considered).".format(repr(op), op.dims,
                                                                                            not_possible_factors,
                                                                                            factor_in))
                        return -1, None
                    op_force = op.op_call
            if len(op.dims.param) > 0 and op.dims.param[0] % factor != 0:
                not_possible_factors.append(factor)
                factor = get_next_larger_dividor(op.dims.param[0], factor, not_possible_factors=not_possible_factors)
                if factor == -1:
                    print("[DOSA:ParallelizeOpClass:ERROR] Unable to find possible split factor for op {}, dims: {}"
                          ", factors tried {}, requested {}.".format(repr(op), op.dims, not_possible_factors,
                                                                     factor_in))
                    return -1, None
                op_force = op.op_call
                necessary = True
    is_possible = True
    util_class = ParallelizeOpClass()
    new_ops_dict = {}
    # TODO: overwrite always legal?
    with_inputs = False
    for oid in orig_brick.ops:
        op = orig_brick.ops[oid]
        if op.op_call not in __ops_possible_to_paralleize__:
            is_possible = False
            print("[DOSA:ParallelizeOpClass:INFO] Operation {} can't be parallelized.".format(op.op_call))
        else:
            att_i = __ops_possible_to_paralleize__.index(op.op_call)
            if with_inputs and not __compatible_with_splitted_input__[att_i]:
                print('[DOSA:ParallelizeOpClass:ERROR] Forced to split input of an operation that is not compatible '
                      'with it ({},{}). STOP.'.format(op.global_op_id, op.op_call))
                exit(1)
            nl = util_class.parallelize(op, factor, with_inputs)
            if __force_to_split_inputs_after_op__[att_i]:
                with_inputs = True
            if nl is None:
                is_possible = False
            else:
                new_ops_dict[oid] = nl
    if not is_possible:
        return -1, None
    if factor_in != factor:
        print("[DOSA:parellizeBrick:INFO] updated split factor to {} for op {}.".format(factor, op_force))
    return factor, new_ops_dict
    # # build bricks
    # new_brick_list = []
    # for i in range(0, factor):
    #     new_brick = ArchBrick()
    #     new_brick.name = orig_brick.name + '_split_{}/{}'.format(i, factor)
    #     new_brick.fn_label = orig_brick.fn_label + '_split_{}/{}'.format(i, factor)
    #     new_brick.tvm_dtype = orig_brick.tvm_dtype
    #     new_brick.used_dtype = orig_brick.used_dtype
    #     new_brick.flops_conv_factor = orig_brick.flops_conv_factor
    #     new_brick.available_osgs = orig_brick.available_osgs
    #     new_brick.possible_osgs = orig_brick.possible_osgs
    #     new_brick.possible_hw_types = orig_brick.possible_hw_types
    #     op_list_new_brick = []
    #     for oid in orig_brick.ops:
    #         op_list_new_brick.append(new_ops_dict[oid][i])
    #     new_brick.reconstruct_from_op_list(op_list_new_brick)
    #     new_brick.orig_brick_object = orig_brick
    #     new_brick.selected_impl_type = orig_brick.selected_impl_type
    #     new_brick.available_contracts = []
    #     new_brick_list.append(new_brick)
    # orig_brick.parallelized_bricks = new_brick_list
    # orig_brick.needs_compute_parallelization = True
    # return factor, orig_brick


class ParallelizeOpClass(object):

    def __init__(self):
        self._method_cache = {}

    def parallelize(self, orig_op, factor, with_inputs):
        if self._method_cache is None:
            self._method_cache = {}
        method_name = orig_op.op_call.split('.')[-1]
        parF = self._method_cache.get(method_name, None)
        if parF is None:
            method = 'parallelize_' + method_name
            # get method or default
            parF = getattr(self, method, self.fallback_parallelize)
            self._method_cache[method_name] = parF
        return parF(orig_op, factor, with_inputs)

    def fallback_parallelize(self, orig_op, factor, with_inputs):
        print('[DOSA:BrickParallelize:ERROR] Attempt to parallelize op {}, but this is not possible. STOP.'
              .format(orig_op.op_call))
        exit(1)

    def parallelize_conv2d(self, orig_op, factor, with_inputs):
        if orig_op.dims.param[0] % factor != 0:
            # TODO: select then next best factor?
            return None
        orig_weights = orig_op.tvm_args['by_position'][1]['ref'].data.numpy()
        new_weights_list = np.split(orig_weights, factor)
        new_ops_list = []
        for i in range(0, factor):
            new_op = ArchOp()
            new_op.name = orig_op.name + '_split_{}of{}'.format(i+1, factor)
            new_op.op_call = orig_op.op_call
            new_op.layer_name = orig_op.layer_name
            new_op.parent_fn = orig_op.parent_fn
            new_op.tvm_dtype = orig_op.tvm_dtype
            new_op.used_dtype = orig_op.used_dtype
            new_op.flops_conv_factor = orig_op.flops_conv_factor
            # FIXME: also adapt tvm_node
            new_op.tvm_node = orig_op.tvm_node
            if not with_inputs:
                new_op.dims.inp = orig_op.dims.inp  # inp stays
                new_op.input_bytes = orig_op.input_bytes
            else:
                new_op.dims.inp = []
                inp_B = np.ceil(get_bitwidth_of_DosaDtype(new_op.used_dtype)/8)
                for dp in range(0, len(orig_op.dims.inp)):
                    d = orig_op.dims.inp[dp]
                    if dp == 1:
                        new_op.dims.inp.append(int(d/factor))
                        inp_B *= int(d/factor)
                    else:
                        new_op.dims.inp.append(d)
                        inp_B *= d
                new_op.input_bytes = int(inp_B)
            new_op.dims.param = []
            param_B = np.ceil(get_bitwidth_of_DosaDtype(new_op.used_dtype)/8)
            # for d in orig_op.dims.param:
            for dp in range(0, len(orig_op.dims.param)):
                d = orig_op.dims.param[dp]
                if dp == 0:
                    new_op.dims.param.append(int(d/factor))
                    param_B *= int(d/factor)
                else:
                    new_op.dims.param.append(d)
                    param_B *= d
            new_op.parameter_bytes = int(param_B)
            new_op.dims.out = []
            out_B = np.ceil(get_bitwidth_of_DosaDtype(new_op.used_dtype)/8)
            # for d in orig_op.dims.out:
            for dp in range(0, len(orig_op.dims.out)):
                d = orig_op.dims.out[dp]
                if dp == 1:
                    new_op.dims.out.append(int(d/factor))
                    out_B *= int(d/factor)
                else:
                    new_op.dims.out.append(d)
                    out_B *= d
            new_op.output_bytes = int(out_B)
            new_op.flops = orig_op.flops/factor
            new_op.oi_stream = new_op.flops / new_op.input_bytes
            new_op.oi_engine = new_op.flops / (new_op.input_bytes + new_op.parameter_bytes)
            new_op.tvm_args = {'calls': [], 'constants': [], 'vars': [], 'else': [], 'by_position': []}
            # input stays
            new_op.tvm_args['by_position'] = [None, None]
            new_op.tvm_args['by_position'][0] = orig_op.tvm_args['by_position'][0].copy()
            orig_var = orig_op.tvm_args['by_position'][1]['node']
            new_op.tvm_args['by_position'][1] = {'pos': 1,
                                                 'node': relay.var(orig_var.name_hint, shape=new_op.dims.param,
                                                                   dtype=new_op.tvm_dtype),
                                                 'ref': relay.const(new_weights_list[i], dtype=new_op.tvm_dtype)}
            # we need new contracts
            new_ops_list.append(new_op)
        return new_ops_list

    def parallelize_bias_add(self, orig_op, factor, with_inputs):
        if orig_op.dims.param[0] % factor != 0:
            # TODO: select then next best factor?
            return None
        orig_bias = orig_op.tvm_args['by_position'][1]['ref'].data.numpy()
        new_bias_list = np.split(orig_bias, factor)
        new_ops_list = []
        for i in range(0, factor):
            new_op = ArchOp()
            new_op.name = orig_op.name + '_split_{}of{}'.format(i+1, factor)
            new_op.op_call = orig_op.op_call
            new_op.layer_name = orig_op.layer_name
            new_op.parent_fn = orig_op.parent_fn
            new_op.tvm_dtype = orig_op.tvm_dtype
            new_op.used_dtype = orig_op.used_dtype
            new_op.flops_conv_factor = orig_op.flops_conv_factor
            # no TVM node --> later # TODO
            if not with_inputs:
                new_op.dims.inp = orig_op.dims.inp  # inp stays
                new_op.input_bytes = orig_op.input_bytes
            else:
                new_op.dims.inp = []
                inp_B = np.ceil(get_bitwidth_of_DosaDtype(new_op.used_dtype)/8)
                for dp in range(0, len(orig_op.dims.inp)):
                    d = orig_op.dims.inp[dp]
                    if dp == 1:
                        new_op.dims.inp.append(int(d/factor))
                        inp_B *= int(d/factor)
                    else:
                        new_op.dims.inp.append(d)
                        inp_B *= d
                new_op.input_bytes = int(inp_B)
            new_op.dims.param = []
            param_B = np.ceil(get_bitwidth_of_DosaDtype(new_op.used_dtype)/8)
            # for d in orig_op.dims.param:
            for dp in range(0, len(orig_op.dims.param)):
                d = orig_op.dims.param[dp]
                if dp == 0:
                    new_op.dims.param.append(int(d/factor))
                    param_B *= int(d/factor)
                else:
                    new_op.dims.param.append(d)
                    param_B *= d
            new_op.parameter_bytes = int(param_B)
            new_op.dims.out = []
            out_B = np.ceil(get_bitwidth_of_DosaDtype(new_op.used_dtype)/8)
            # for d in orig_op.dims.out:
            for dp in range(0, len(orig_op.dims.out)):
                d = orig_op.dims.out[dp]
                if dp == 1:
                    new_op.dims.out.append(int(d/factor))
                    out_B *= int(d/factor)
                else:
                    new_op.dims.out.append(d)
                    out_B *= d
            new_op.output_bytes = int(out_B)
            new_op.flops = orig_op.flops/factor
            new_op.oi_stream = new_op.flops / new_op.input_bytes
            new_op.oi_engine = new_op.flops / (new_op.input_bytes + new_op.parameter_bytes)
            new_op.tvm_args = {'calls': [], 'constants': [], 'vars': [], 'else': [], 'by_position': []}
            # input stays
            new_op.tvm_args['by_position'] = [None, None]
            new_op.tvm_args['by_position'][0] = orig_op.tvm_args['by_position'][0].copy()
            orig_var = orig_op.tvm_args['by_position'][1]['node']
            new_op.tvm_args['by_position'][1] = {'pos': 1,
                                                 'node': relay.var(orig_var.name_hint, shape=new_op.dims.param,
                                                                   dtype=new_op.tvm_dtype),
                                                 'ref': relay.const(new_bias_list[i], dtype=new_op.tvm_dtype)}
            new_ops_list.append(new_op)
        return new_ops_list

    def parallelize_tanh(self, orig_op, factor, with_inputs):
        new_ops_list = []
        for i in range(0, factor):
            new_op = ArchOp()
            new_op.name = orig_op.name + '_split_{}of{}'.format(i+1, factor)
            new_op.op_call = orig_op.op_call
            new_op.layer_name = orig_op.layer_name
            new_op.parent_fn = orig_op.parent_fn
            new_op.tvm_dtype = orig_op.tvm_dtype
            new_op.used_dtype = orig_op.used_dtype
            new_op.flops_conv_factor = orig_op.flops_conv_factor
            # no TVM node --> later # TODO
            if not with_inputs:
                new_op.dims.inp = orig_op.dims.inp  # inp stays
                new_op.input_bytes = orig_op.input_bytes
            else:
                new_op.dims.inp = []
                inp_B = np.ceil(get_bitwidth_of_DosaDtype(new_op.used_dtype)/8)
                for dp in range(0, len(orig_op.dims.inp)):
                    d = orig_op.dims.inp[dp]
                    if dp == 1:
                        new_op.dims.inp.append(int(d/factor))
                        inp_B *= int(d/factor)
                    else:
                        new_op.dims.inp.append(d)
                        inp_B *= d
                new_op.input_bytes = int(inp_B)
            new_op.dims.param = []
            new_op.parameter_bytes = 0
            new_op.dims.out = []
            out_B = np.ceil(get_bitwidth_of_DosaDtype(new_op.used_dtype)/8)
            # for d in orig_op.dims.out:
            for dp in range(0, len(orig_op.dims.out)):
                d = orig_op.dims.out[dp]
                if dp == 1:
                    new_op.dims.out.append(int(d/factor))
                    out_B *= int(d/factor)
                else:
                    new_op.dims.out.append(d)
                    out_B *= d
            new_op.output_bytes = int(out_B)
            new_op.flops = orig_op.flops/factor
            new_op.oi_stream = new_op.flops / new_op.input_bytes
            new_op.oi_engine = new_op.flops / (new_op.input_bytes + new_op.parameter_bytes)
            new_op.tvm_args = {'calls': [], 'constants': [], 'vars': [], 'else': [], 'by_position': []}
            # input stays
            new_op.tvm_args['by_position'] = [None]
            new_op.tvm_args['by_position'][0] = orig_op.tvm_args['by_position'][0].copy()
            # only one arg
            new_ops_list.append(new_op)
        return new_ops_list

    def parallelize_relu(self, orig_op, factor, with_inputs):
        new_ops_list = []
        for i in range(0, factor):
            new_op = ArchOp()
            new_op.name = orig_op.name + '_split_{}of{}'.format(i+1, factor)
            new_op.op_call = orig_op.op_call
            new_op.layer_name = orig_op.layer_name
            new_op.parent_fn = orig_op.parent_fn
            new_op.tvm_dtype = orig_op.tvm_dtype
            new_op.used_dtype = orig_op.used_dtype
            new_op.flops_conv_factor = orig_op.flops_conv_factor
            # no TVM node --> later # TODO
            if not with_inputs:
                new_op.dims.inp = orig_op.dims.inp  # inp stays
                new_op.input_bytes = orig_op.input_bytes
            else:
                new_op.dims.inp = []
                inp_B = np.ceil(get_bitwidth_of_DosaDtype(new_op.used_dtype)/8)
                for dp in range(0, len(orig_op.dims.inp)):
                    d = orig_op.dims.inp[dp]
                    if dp == 1:
                        new_op.dims.inp.append(int(d/factor))
                        inp_B *= int(d/factor)
                    else:
                        new_op.dims.inp.append(d)
                        inp_B *= d
                new_op.input_bytes = int(inp_B)
            new_op.dims.param = []
            new_op.parameter_bytes = 0
            new_op.dims.out = []
            out_B = np.ceil(get_bitwidth_of_DosaDtype(new_op.used_dtype)/8)
            # for d in orig_op.dims.out:
            for dp in range(0, len(orig_op.dims.out)):
                d = orig_op.dims.out[dp]
                if dp == 1:
                    new_op.dims.out.append(int(d/factor))
                    out_B *= int(d/factor)
                else:
                    new_op.dims.out.append(d)
                    out_B *= d
            new_op.output_bytes = int(out_B)
            new_op.flops = orig_op.flops/factor
            new_op.oi_stream = new_op.flops / new_op.input_bytes
            new_op.oi_engine = new_op.flops / (new_op.input_bytes + new_op.parameter_bytes)
            new_op.tvm_args = {'calls': [], 'constants': [], 'vars': [], 'else': [], 'by_position': []}
            # input stays
            new_op.tvm_args['by_position'] = [None]
            new_op.tvm_args['by_position'][0] = orig_op.tvm_args['by_position'][0].copy()
            # only one arg
            new_ops_list.append(new_op)
        return new_ops_list

    def parallelize_max_pool2d(self, orig_op, factor, with_inputs):
        new_ops_list = []
        for i in range(0, factor):
            new_op = ArchOp()
            new_op.name = orig_op.name + '_split_{}of{}'.format(i+1, factor)
            new_op.op_call = orig_op.op_call
            new_op.layer_name = orig_op.layer_name
            new_op.parent_fn = orig_op.parent_fn
            new_op.tvm_dtype = orig_op.tvm_dtype
            new_op.used_dtype = orig_op.used_dtype
            new_op.flops_conv_factor = orig_op.flops_conv_factor
            # FIXME: also adapt tvm_node
            new_op.tvm_node = orig_op.tvm_node
            if not with_inputs:
                new_op.dims.inp = orig_op.dims.inp  # inp stays
                new_op.input_bytes = orig_op.input_bytes
                new_op.dims.out = orig_op.dims.out
                new_op.output_bytes = orig_op.output_bytes
            else:
                new_op.dims.inp = []
                inp_B = np.ceil(get_bitwidth_of_DosaDtype(new_op.used_dtype)/8)
                for dp in range(0, len(orig_op.dims.inp)):
                    d = orig_op.dims.inp[dp]
                    if dp == 1:
                        new_op.dims.inp.append(int(d/factor))
                        inp_B *= int(d/factor)
                    else:
                        new_op.dims.inp.append(d)
                        inp_B *= d
                new_op.input_bytes = int(inp_B)
                new_op.dims.out = []
                out_B = np.ceil(get_bitwidth_of_DosaDtype(new_op.used_dtype)/8)
                # for d in orig_op.dims.out:
                for dp in range(0, len(orig_op.dims.out)):
                    d = orig_op.dims.out[dp]
                    if dp == 1:
                        new_op.dims.out.append(int(d/factor))
                        out_B *= int(d/factor)
                    else:
                        new_op.dims.out.append(d)
                        out_B *= d
                new_op.output_bytes = int(out_B)
            new_op.dims.param = []
            new_op.parameter_bytes = 0
            new_op.flops = orig_op.flops/factor
            new_op.oi_stream = new_op.flops / new_op.input_bytes
            new_op.oi_engine = new_op.flops / (new_op.input_bytes + new_op.parameter_bytes)
            new_op.tvm_args = {'calls': [], 'constants': [], 'vars': [], 'else': [], 'by_position': []}
            # input stays
            new_op.tvm_args['by_position'] = [None]
            new_op.tvm_args['by_position'][0] = orig_op.tvm_args['by_position'][0].copy()
            # only one arg
            new_ops_list.append(new_op)
        return new_ops_list

    def parallelize_batch_flatten(self, orig_op, factor, with_inputs):
        # we usually ignore tlasts...-> doesn't matter
        # tkeeps are more important
        new_ops_list = []
        for i in range(0, factor):
            new_op = ArchOp()
            new_op.name = orig_op.name + '_split_{}of{}'.format(i+1, factor)
            new_op.op_call = orig_op.op_call
            new_op.layer_name = orig_op.layer_name
            new_op.parent_fn = orig_op.parent_fn
            new_op.tvm_dtype = orig_op.tvm_dtype
            new_op.used_dtype = orig_op.used_dtype
            new_op.flops_conv_factor = orig_op.flops_conv_factor
            # no TVM node --> later # TODO
            if not with_inputs:
                new_op.dims.inp = orig_op.dims.inp  # inp stays
                new_op.input_bytes = orig_op.input_bytes
            else:
                new_op.dims.inp = []
                inp_B = np.ceil(get_bitwidth_of_DosaDtype(new_op.used_dtype)/8)
                for dp in range(0, len(orig_op.dims.inp)):
                    d = orig_op.dims.inp[dp]
                    if dp == 1:
                        new_op.dims.inp.append(int(d/factor))
                        inp_B *= int(d/factor)
                    else:
                        new_op.dims.inp.append(d)
                        inp_B *= d
                new_op.input_bytes = int(inp_B)
            new_op.dims.param = []
            new_op.parameter_bytes = 0
            new_op.dims.out = []
            out_B = np.ceil(get_bitwidth_of_DosaDtype(new_op.used_dtype)/8)
            # for d in orig_op.dims.out:
            for dp in range(0, len(orig_op.dims.out)):
                d = orig_op.dims.out[dp]
                if dp == 1:
                    new_op.dims.out.append(int(d/factor))
                    out_B *= int(d/factor)
                else:
                    new_op.dims.out.append(d)
                    out_B *= d
            new_op.output_bytes = int(out_B)
            new_op.flops = orig_op.flops/factor
            new_op.oi_stream = new_op.flops / new_op.input_bytes
            new_op.oi_engine = new_op.flops / (new_op.input_bytes + new_op.parameter_bytes)
            new_op.tvm_args = {'calls': [], 'constants': [], 'vars': [], 'else': [], 'by_position': []}
            # input stays
            new_op.tvm_args['by_position'] = [None]
            new_op.tvm_args['by_position'][0] = orig_op.tvm_args['by_position'][0].copy()
            # only one arg
            new_ops_list.append(new_op)
        return new_ops_list

