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

from dimidium.lib.util import replace_deep


__bits_per_byte__ = 8


@relay.transform.function_pass(opt_level=1)
class OiPipeline:
    """Relay pass to calculate OIs"""

    def __init__(self, size_t, oiCalc):
        self.oi_results = []
        self.bw_results = []
        self.data_per_layer = {}
        self.size_t = size_t
        self.size_b = math.ceil(size_t/__bits_per_byte__)
        self.bw_layer_cnt = 0
        self.oiCalc = oiCalc
        self.fn_cnt = 0
        self.cur_fstr = "none"
        self.fn_dict = {}
        self.oi_fused_wise = {}
        self.oi_main_view = {}
        self.fn_call_cnts = {}

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
        repdict = {'FN_0000': 'fn_main'}
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
                if obj.fn_cnt > 9999:
                    print("[DOSA:OICALC:ERROR] fn name overflow occurred!")
                my_fstr = "{:04}".format(my_cnt)
                my_name = 'FN_{}'.format(my_fstr)
                obj.cur_fstr = my_name
                obj.oi_fused_wise[obj.cur_fstr] = []
                obj.fn_call_cnts[obj.cur_fstr] = 0

                if my_cnt == 0:
                    my_name = '(input)'
                my_hash = str(hash(func))
                obj.fn_dict[my_hash] = my_name

                if hasattr(func, 'params'):
                    bw_tmp = obj.size_b
                    for p in func.params:
                        # TODO use p.checked_type.dtype instead of size_b?
                        bw_type = None
                        if hasattr(p, 'checked_type'):
                            bw_type = p.checked_type
                        elif hasattr(p, 'type_annotation'):
                            bw_type = p.type_annotation
                        if bw_type is not None:
                            for d in bw_type.shape:
                                bw_tmp *= int(d)
                        if hasattr(p, 'name_hint'):   # necessary?
                            hint = p.name_hint
                my_layer_num = obj.bw_layer_cnt
                obj.bw_layer_cnt += 1
                res = {'num_layer': my_layer_num, 'name': hint, 'bw_total_B': bw_tmp,
                       'bw_data_B': bw_tmp, 'bw_param_B': 0}
                obj.bw_results.append(res)
                istr = "{:06}".format(my_layer_num)
                dpl = {'name': my_name, 'cmpl': 0, 'uinp': 0, 'flop': 0, 'parB': 0, 'inpB': bw_tmp, 'outB': bw_tmp,
                       'layer': istr}
                obj.data_per_layer[istr] = dpl
                # visit body
                self.visit(func.body)
                # calculate output bw
                if my_cnt == 0:
                    my_layer_num2 = obj.bw_layer_cnt
                    obj.bw_layer_cnt += 1
                    bw_tmp = "unknown"
                    hint = "func_return"
                    if hasattr(func, 'ret_type'):
                        bw_tmp = obj.size_b
                        for d in func.ret_type.shape:
                            bw_tmp *= int(d)
                        if hasattr(func.ret_type, 'name_hint'):
                            hint = func.ret_type.name_hint
                    res2 = {'num_layer': my_layer_num2, 'name': hint, 'bw_total_B': bw_tmp,
                            'bw_data_B': bw_tmp, 'bw_param_B': 0}
                    obj.bw_results.append(res2)
                    istr = "{:06}".format(my_layer_num2)
                    dpl2 = {'name': '(output)', 'cmpl': 0, 'uinp': 0, 'flop': 0, 'parB': 0, 'inpB': bw_tmp, 'outB': bw_tmp,
                            'layer': istr}
                    obj.data_per_layer[istr] = dpl2

            def visit_call(self, call):
                self.visit(call.op)  # necessary?
                for a in call.args:
                    self.visit(a)

                # post order processing
                my_layer_num = obj.bw_layer_cnt
                obj.bw_layer_cnt += 1
                if obj.bw_layer_cnt > 999999:
                    print("[DOSA:OICALC:ERROR] layer count overflow occurred!")
                function_call = False
                if type(call.op) is relay.function.Function:
                    f_hash = str(hash(call.op))
                    my_name = obj.fn_dict[f_hash]
                    op_name = my_name
                    function_call = True
                else:
                    my_name = str(call.op.name)
                    op_name = my_name
                    if obj.fn_cnt > 1:
                        my_name = "{}_{}".format(obj.cur_fstr, str(call.op.name))
                bw_data_B = 0
                bw_param_B = 0
                data_dim = []
                param_dim = []
                for a in call.args:
                    # TODO use checked_type.dtype instead of size_b?
                    bw_tmp = obj.size_b
                    cur_dims = []
                    for d in a.checked_type.shape:
                        bw_tmp *= int(d)
                        cur_dims.append(int(d))
                    if type(a) is tvm.relay.expr.Constant:
                        # parameters
                        param_dim.append(cur_dims)
                        bw_param_B += bw_tmp
                    else:
                        # data
                        data_dim.append(cur_dims)
                        bw_data_B += bw_tmp
                bw_total_B = bw_param_B + bw_data_B
                out_bw = 0
                out_dim = []
                if hasattr(call, 'checked_type'):
                    out_bw = obj.size_b
                    cur_dims = []
                    for d in call.checked_type.shape:
                        out_bw *= int(d)
                        cur_dims.append(int(d))
                    out_dim.append(cur_dims)

                attrs = None
                if hasattr(call, 'attrs'):
                    attrs = call.attrs

                if not function_call:
                    oi_cmpl, oi_uinp, flop_total = obj.oiCalc.calc(call.op.name, data_dim, param_dim, out_dim, attrs, obj.size_b)
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
                istr = "{:06}".format(my_layer_num)
                dpl = {'name': my_name, 'cmpl': oi_cmpl, 'uinp': oi_uinp, 'flop': flop_total, 'parB': bw_param_B,
                       'inpB': bw_data_B, 'outB': out_bw, 'layer': istr, 'fn': obj.cur_fstr, 'op': op_name}
                obj.data_per_layer[istr] = dpl
                obj.oi_fused_wise[obj.cur_fstr].append(dpl)
                if function_call:
                    dpl['layer'] = name_summary
                    obj.oi_main_view[my_name] = dpl
                    obj.fn_call_cnts[my_name] += 1

        return OiVisitor().visit(func)

