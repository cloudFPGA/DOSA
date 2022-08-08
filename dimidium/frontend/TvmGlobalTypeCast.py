#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Feb 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *       Tvm Pass to correct types imported from ONNX
#  *
#  *
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay


@relay.transform.function_pass(opt_level=1)
class CorrectionPipeline:
    """Simple tvm pass to correct types imported from ONNX."""

    def __init__(self, input_name, var_type, constant_type):
        self.input_name = input_name
        self.var_type = var_type
        self.constant_type = constant_type

    def transform_function(self, func, mod, ctx):
        obj = self

        class GlobalCastDtypes(tvm.relay.ExprMutator):
            def visit_constant(self, c):
                # new_c = relay.cast(c, obj.constant_type)
                new_c = relay.const(c.data.numpy().astype(obj.constant_type), dtype=obj.constant_type)
                return new_c

            def visit_var(self, v):
                if v.name_hint == obj.input_name:
                    # new_v = relay.cast(v, obj.var_type)
                    # new_v = v
                    # new_v.type_annotation.dtype = obj.var_type
                    new_v = relay.var(v.name_hint, shape=v.type_annotation.concrete_shape, dtype=obj.var_type)
                    return new_v
                return v

            def visit_function(self, f):
                new_params = []
                for p in f.params:
                    new_params.append(self.visit(p))
                new_body = self.visit(f.body)
                new_f = relay.function.Function(params=new_params, body=new_body)
                return new_f

            # def visit_call(self, call):
            #     new_fn = self.visit(call.op)
            #     new_args = [self.visit(arg) for arg in call.args]
            #     if hasattr(call.attrs, 'out_dtype') and len(call.attrs.out_dtype) > 0:
            #         call.attrs.out_dtype = obj.var_type
            #         call.attrs.__setattr__('out_dtype', 'int8')
            #     # print(call.attrs)
            #     return relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)

            # def visit_op(self, o):
            #     return o

        return GlobalCastDtypes().visit(func)

