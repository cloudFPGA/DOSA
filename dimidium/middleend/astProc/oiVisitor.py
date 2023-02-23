#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Jun 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Visitor for calculating the OI for each call
#  *
#  *

import numpy as np
import tvm
import tvm.relay as relay
import math
from tvm.ir.type import TupleType
from tvm.ir.tensor_type import TensorType

from dimidium.lib.util import replace_deep, dtype_to_size_b, bit_to_dtype
from dimidium.lib.units import config_bits_per_byte
import dimidium.lib.singleton as dosa_singleton


oiV_fn_main_str = 'fn_main'
oiV_input_str = '(input)'
oiV_output_str = '(output)'
oiV_func_str = '(function)'
oiV_func_call_str = '(function_call)'


@relay.transform.function_pass(opt_level=1)
class OiPipeline:
    """Relay pass to calculate OIs"""

    def __init__(self, fallback_size_t, oiCalc):
        self.oi_results = []
        self.bw_results = []
        self.data_per_layer = {}
        # self.datatypes_per_layer = {}
        self.size_t = fallback_size_t
        self.default_size_b = math.ceil(self.size_t/config_bits_per_byte)
        self.default_dtype = bit_to_dtype(fallback_size_t)
        self.bw_layer_cnt = 0
        self.oiCalc = oiCalc
        self.fn_cnt = 0
        self.cur_fstr = "none"
        self.fn_dict = {}
        self.oi_fused_wise = {}
        self.oi_main_view = {}
        self.fn_call_cnts = {}
        self.node_cnt = 0
        self.tvm_nodes = {}
        self.func_call_args = {}
        self.cur_f_args = []

    def get_oi_results(self):
        return self.oi_results

    def get_bw_results(self):
        return self.bw_results

    def get_data_per_layer(self):
        return self.data_per_layer

    def get_oi_fused_wise(self):
        return self.oi_fused_wise

    def get_oi_main_view(self):
        return self.oi_main_view

    def get_fn_call_cnts(self):
        return self.fn_call_cnts

    def reorder_fn_calls(self):
        tl = self.fn_cnt
        repdict = {'FN_0000': oiV_fn_main_str}
        for i in range(1, tl):
            o_str = "{:04}".format(i)
            n_str = "{:04}".format(tl-i)
            ok = 'FN_{}'.format(o_str)
            nk = 'fn_{}'.format(n_str)
            repdict[ok] = nk
        new_oi_results = replace_deep(self.oi_results, repdict)
        new_bw_results = replace_deep(self.bw_results, repdict)
        new_dpl = replace_deep(self.data_per_layer, repdict)
        new_oi_fused_wise = replace_deep(self.oi_fused_wise, repdict)
        new_oi_main_view = replace_deep(self.oi_main_view, repdict)
        new_fn_stats = replace_deep(self.fn_call_cnts, repdict)
        self.oi_results = new_oi_results
        self.bw_results = new_bw_results
        self.data_per_layer = new_dpl
        self.oi_fused_wise = new_oi_fused_wise
        self.oi_main_view = new_oi_main_view
        self.fn_call_cnts = new_fn_stats
        # no need to reorder tvm nodes & func call args

    def get_tvm_nodes(self):
        return self.tvm_nodes

    def get_call_args(self):
        return self.func_call_args

    # This function can define a pass.
    def transform_function(self, func, mod, ctx):
        obj = self

        # class OiVisitor(tvm.relay.ExprMutator):  # we don't want to modify it...
        class OiVisitor(tvm.relay.ExprVisitor):

            def visit_function(self, func):
                # calculate input bw
                bw_tmp = "unknown"
                hint = "no_hint"
                my_cnt = obj.fn_cnt
                obj.fn_cnt += 1
                my_node_id = obj.node_cnt
                obj.node_cnt += 1
                if obj.fn_cnt > 9999:
                    print("[DOSA:OICALC:ERROR] fn name overflow occurred!")
                my_fstr = "{:04}".format(my_cnt)
                my_name = 'FN_{}'.format(my_fstr)
                old_fstr = obj.cur_fstr
                obj.cur_fstr = my_name
                obj.oi_fused_wise[obj.cur_fstr] = []
                obj.fn_call_cnts[obj.cur_fstr] = 0

                fn_name = my_name
                if my_cnt == 0:
                    my_name = oiV_input_str
                    fn_name = oiV_fn_main_str
                my_hash = str(hash(func))
                obj.fn_dict[my_hash] = my_name

                # used_dtype = obj.default_dtype
                used_dtype = None
                if hasattr(func, 'params'):
                    bw_tmp = obj.default_size_b
                    for p in func.params:
                        # TODO use p.checked_type.dtype instead of size_b?
                        bw_type = None
                        if hasattr(p, 'checked_type'):
                            if type(p.checked_type) is TensorType:
                                bw_type = p.checked_type
                                bw_tmp = dtype_to_size_b(p.checked_type.dtype)
                                used_dtype = p.checked_type.dtype
                            elif type(p.checked_type) is TupleType:
                                # TODO: necessary?
                                # take the first one, should be similar
                                bw_type = p.checked_type.fields[0]
                                bw_tmp = dtype_to_size_b(p.checked_type.fields[0].dtype)
                                used_dtype = p.checked_type.fields[0].dtype
                        elif hasattr(p, 'type_annotation'):
                            bw_type = p.type_annotation
                            bw_tmp = dtype_to_size_b(p.type_annotation.dtype)
                            used_dtype = p.type_annotation.dtype
                        if bw_type is not None:
                            for d in bw_type.shape:
                                bw_tmp *= int(d)
                        if hasattr(p, 'name_hint'):   # necessary?
                            hint = p.name_hint
                my_layer_num = obj.bw_layer_cnt
                istr = "{:06}".format(my_layer_num)
                orig_dtype = used_dtype
                if dosa_singleton.config.quant.overwrite_imported_dtypes:
                    # TODO: support different dtypes for weights and activations
                    print(f"[DOSA:OICALC:INFO] overwriting dtypes of function {my_name} with "
                          f"{dosa_singleton.config.quant.activation_dtype} (orig: {used_dtype}).")
                    used_dtype = dosa_singleton.config.quant.activation_dtype
                # obj.datatypes_per_layer[istr] = used_dtype
                obj.bw_layer_cnt += 1
                res = {'num_layer': my_layer_num, 'name': hint, 'bw_total_B': bw_tmp,
                       'bw_data_B': bw_tmp, 'bw_param_B': 0}
                obj.bw_results.append(res)
                dpl = {'name': my_name, 'cmpl': 0, 'uinp': 0, 'flop': 0, 'parB': 0, 'inpB': bw_tmp, 'outB': bw_tmp,
                       'layer': istr, 'fn': fn_name, 'op': oiV_func_str, 'dtype': used_dtype, 'tid': my_node_id,
                       'orig_dtype': orig_dtype}
                obj.data_per_layer[istr] = dpl
                obj.tvm_nodes[my_node_id] = func
                # visit body
                self.visit(func.body)
                # reset fstr
                obj.cur_fstr = old_fstr
                # in case of main func, calculate output bw
                if my_cnt == 0:
                    my_layer_num2 = obj.bw_layer_cnt
                    obj.bw_layer_cnt += 1
                    bw_tmp = "unknown"
                    hint = "func_return"
                    if hasattr(func, 'ret_type'):
                        bw_tmp = dtype_to_size_b(used_dtype)
                        for d in func.ret_type.shape:
                            bw_tmp *= int(d)
                        if hasattr(func.ret_type, 'name_hint'):
                            hint = func.ret_type.name_hint
                    res2 = {'num_layer': my_layer_num2, 'name': hint, 'bw_total_B': bw_tmp,
                            'bw_data_B': bw_tmp, 'bw_param_B': 0}
                    obj.bw_results.append(res2)
                    istr = "{:06}".format(my_layer_num2)
                    dpl2 = {'name': oiV_output_str, 'cmpl': 0, 'uinp': 0, 'flop': 0, 'parB': 0, 'inpB': bw_tmp, 'outB': bw_tmp,
                            'layer': istr,  'fn': fn_name, 'op': oiV_func_str, 'dtype': used_dtype, 'tid': my_node_id,
                            'orig_dtype': orig_dtype}
                    obj.data_per_layer[istr] = dpl2

            def visit_call(self, call):
                function_call = False
                if type(call.op) is relay.function.Function:
                    function_call = True
                    cur_arg_list = []
                    for a in call.args:
                        cur_arg_list.append(a)
                    obj.cur_f_args = cur_arg_list

                self.visit(call.op)
                for a in call.args:
                    self.visit(a)

                # post order processing
                my_layer_num = obj.bw_layer_cnt
                obj.bw_layer_cnt += 1
                istr = "{:06}".format(my_layer_num)
                if obj.bw_layer_cnt > 999999:
                    print("[DOSA:OICALC:ERROR] layer count overflow occurred!")
                if type(call.op) is relay.function.Function:
                    f_hash = str(hash(call.op))
                    my_name = obj.fn_dict[f_hash]
                    # op_name = my_name
                    op_name = oiV_func_call_str
                else:
                    my_name = str(call.op.name)
                    op_name = my_name
                    if obj.fn_cnt > 1:
                        my_name = "{}_{}".format(obj.cur_fstr, str(call.op.name))
                my_node_id = obj.node_cnt
                obj.node_cnt += 1

                bw_data_B = 0
                bw_param_B = 0
                data_dim = []
                param_dim = []
                used_dtype = None
                # used_dtype = obj.datatypes_per_layer[istr]
                orginal_dtype = None
                if dosa_singleton.config.quant.overwrite_imported_dtypes:
                    used_dtype = dosa_singleton.config.quant.activation_dtype
                call_args_dict = {'calls': [], 'constants': [], 'vars': [], 'else': [], 'by_position': []}
                arg_pos = 0
                for a in call.args:
                    # bw_tmp = obj.size_b
                    # bw_tmp = dtype_to_size_b(a.checked_type.dtype)
                    original_dtype = a.checked_type.dtype
                    if used_dtype is None:
                        used_dtype = a.checked_type.dtype
                    elif used_dtype != a.checked_type.dtype and \
                            not dosa_singleton.config.quant.overwrite_imported_dtypes:
                        # TODO: support different dtypes for weights and activations
                        print("[DOSA:OICALC:WARNING] different dtypes within one operation ({}). Ignoring."
                              .format(op_name))
                    bw_tmp = dtype_to_size_b(used_dtype)
                    cur_dims = []
                    for d in a.checked_type.shape:
                        bw_tmp *= int(d)
                        cur_dims.append(int(d))
                    if (type(a) is tvm.relay.expr.Constant) or \
                            ((isinstance(a, tvm.relay.expr.Var) or isinstance(a, tvm.relay.expr.GlobalVar)) and
                             (not function_call and len(obj.cur_f_args) > 0) and
                             (type(obj.cur_f_args[0]) is tvm.relay.expr.Constant)
                            ):
                        # parameters
                        param_dim.append(cur_dims)
                        bw_param_B += bw_tmp
                    else:
                        # data
                        data_dim.append(cur_dims)
                        bw_data_B += bw_tmp

                    arg_dict = {'pos': arg_pos, 'node': a, 'ref': None}
                    call_args_dict['by_position'].append(arg_dict)
                    arg_pos += 1
                    if isinstance(a, tvm.relay.expr.Constant):
                        call_args_dict['constants'].append(arg_dict)
                    elif isinstance(a, tvm.relay.expr.Call):
                        call_args_dict['calls'].append(arg_dict)
                    elif isinstance(a, tvm.relay.expr.Var) or isinstance(a, tvm.relay.expr.GlobalVar):
                        if not function_call and len(obj.cur_f_args) > 0:
                            # # consume in reverse order
                            # ref_arg = obj.cur_f_args[-1]
                            # del obj.cur_f_args[-1]
                            # consume in order
                            ref_arg = obj.cur_f_args[0]
                            del obj.cur_f_args[0]
                            arg_dict['ref'] = ref_arg
                        call_args_dict['vars'].append(arg_dict)
                    else:
                        call_args_dict['else'].append(arg_dict)
                bw_total_B = bw_param_B + bw_data_B
                out_bw = 0
                out_dim = []
                if hasattr(call, 'checked_type'):
                    # out_bw = obj.size_b
                    out_bw = dtype_to_size_b(call.checked_type.dtype)
                    cur_dims = []
                    for d in call.checked_type.shape:
                        out_bw *= int(d)
                        cur_dims.append(int(d))
                    out_dim.append(cur_dims)

                attrs = None
                if hasattr(call, 'attrs'):
                    attrs = call.attrs

                # correct param dims
                # TODO: NO, this is invalid...e.g. 'add' call
                # if len(data_dim) == 2 and len(param_dim) == 0:
                #     param_dim = data_dim[1]

                if not function_call:
                    oi_cmpl, oi_uinp, flop_total = obj.oiCalc.calc(call.op.name, data_dim, param_dim, out_dim, attrs,
                                                                   dtype_to_size_b(used_dtype))
                else:
                    flop_total = 0
                    name_sum_t = "fn("
                    for e in obj.oi_fused_wise[my_name]:
                        flop_total += e['flop']
                        tmp_name = e['op'].split('.')[-1]
                        name_sum_t += tmp_name + ", "
                    name_summary = name_sum_t[:-2] + ")"
                    oi_cmpl = float(flop_total) / bw_total_B
                    oi_uinp = float(flop_total) / bw_data_B

                resBw = {'num_layer': my_layer_num, 'name': my_name, 'bw_total_B': bw_total_B,
                         'bw_data_B': bw_data_B, 'bw_param_B': bw_param_B}
                obj.bw_results.append(resBw)
                resOi = {'num_layer': my_layer_num, 'name': my_name, 'oi_cmpl': oi_cmpl, 'oi_uinp': oi_uinp}
                obj.oi_results.append(resOi)
                dpl = {'name': my_name, 'cmpl': oi_cmpl, 'uinp': oi_uinp, 'flop': flop_total, 'parB': bw_param_B,
                       'inpB': bw_data_B, 'outB': out_bw, 'layer': istr, 'fn': obj.cur_fstr, 'op': op_name,
                       'dtype': used_dtype, 'tid': my_node_id, 'dims': {},
                       'orig_dtype': original_dtype}
                dpl['dims']['inp'] = data_dim
                dpl['dims']['param'] = param_dim
                dpl['dims']['out'] = out_dim
                obj.data_per_layer[istr] = dpl
                obj.tvm_nodes[my_node_id] = call
                obj.func_call_args[my_node_id] = call_args_dict
                obj.oi_fused_wise[obj.cur_fstr].append(dpl)
                if function_call:
                    dpl['layer'] = name_summary
                    obj.oi_main_view[my_name] = dpl
                    obj.fn_call_cnts[my_name] += 1

        return OiVisitor().visit(func)

