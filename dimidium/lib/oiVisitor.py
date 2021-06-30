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

__bits_per_byte__ = 8


@relay.transform.function_pass(opt_level=1)
class OiPipeline:
    """Relay pass to calculate OIs"""

    def __init__(self, size_t, oiCalc):
        self.oi_results = []
        self.bw_results = []
        self.size_t = size_t
        self.size_b = math.ceil(size_t/__bits_per_byte__)
        self.bw_layer_cnt = 0
        self.oiCalc = oiCalc

    def get_oi_results(self):
        return self.oi_results

    def get_bw_results(self):
        return self.bw_results

    # This function can define a pass.
    def transform_function(self, func, mod, ctx):
        obj = self

        # class OiVisitor(tvm.relay.ExprMutator):  # we don't want to modify it...
        class OiVisitor(tvm.relay.ExprVisitor):

            def visit_function(self, func):
                # calculate input bw
                bw_tmp = "unknown"
                hint = "no_hint"
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
                                bw_tmp *= d
                        if hasattr(p, 'name_hint'):   # necessary?
                            hint = p.name_hint
                res = {'num_layer': obj.bw_layer_cnt, 'name': hint, 'bw_total_B': bw_tmp,
                         'bw_data_B': bw_tmp, 'bw_param_B': 0}
                obj.bw_layer_cnt += 1
                obj.bw_results.append(res)
                # visit body
                self.visit(func.body)
                # calculate output bw
                bw_tmp = "unknown"
                hint = "func_return"
                if hasattr(func, 'ret_type'):
                    bw_tmp = obj.size_b
                    for d in func.ret_type.shape:
                        bw_tmp *= d
                    if hasattr(func.ret_type, 'name_hint'):
                        hint = func.ret_type.name_hint
                res2 = {'num_layer': obj.bw_layer_cnt, 'name': hint, 'bw_total_B': bw_tmp,
                       'bw_data_B': bw_tmp, 'bw_param_B': 0}
                obj.bw_layer_cnt += 1
                obj.bw_results.append(res2)

            def visit_call(self, call):
                self.visit(call.op)  # necessary?
                for a in call.args:
                    self.visit(a)
                # post order processing
                my_layer_num = obj.bw_layer_cnt
                obj.bw_layer_cnt += 1
                bw_data_B = 0
                bw_param_B = 0
                data_dim = []
                param_dim = []
                for a in call.args:
                    # TODO use checked_type.dtype instead of size_b?
                    bw_tmp = obj.size_b
                    cur_dims = []
                    for d in a.checked_type.shape:
                        bw_tmp *= d
                        cur_dims.append(d)
                    if type(a, tvm.relay.expr.Constant):
                        # parameters
                        param_dim.append(cur_dims)
                        bw_param_B += bw_tmp
                    else:
                        # data
                        data_dim.append(cur_dims)
                        bw_data_B += bw_tmp
                bw_total_B = bw_param_B + bw_data_B

                attrs = None
                if hasattr(call, 'attrs'):
                    attrs = call.attrs
                oiB = obj.oiCalc.calc(call.op.name, data_dim, param_dim, attrs, obj.size_b)

                resBw = {'num_layer': my_layer_num, 'name': call.op.name, 'bw_total_B': bw_total_B,
                         'bw_data_B': bw_data_B, 'bw_param_B': bw_param_B}
                obj.bw_results.append(resBw)

        return OiVisitor().visit(func)

