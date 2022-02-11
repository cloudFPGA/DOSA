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
                new_c = relay.const(c.data.numpy().astype('int8'), dtype='int8')
                return new_c

            def visit_var(self, v):
                if v.name_hint == obj.input_name:
                    # new_v = relay.cast(v, obj.var_type)
                    # new_v = v
                    # new_v.type_annotation.dtype = obj.var_type
                    new_v = relay.var(v.name_hint, shape=v.type_annotation.concrete_shape, dtype='int8')
                    return new_v
                return v

        return GlobalCastDtypes().visit(func)

