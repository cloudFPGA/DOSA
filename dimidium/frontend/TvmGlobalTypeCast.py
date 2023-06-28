#  /*******************************************************************************
#   * Copyright 2019 -- 2023 IBM Corporation
#   *
#   * Licensed under the Apache License, Version 2.0 (the "License");
#   * you may not use this file except in compliance with the License.
#   * You may obtain a copy of the License at
#   *
#   *     http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS,
#   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   * See the License for the specific language governing permissions and
#   * limitations under the License.
#  *******************************************************************************/
#

#  *
#  *                       cloudFPGA
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
from tvm.ir.tensor_type import TensorType


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
                new_params = [self.visit(p) for p in f.params]
                if hasattr(f, 'ret_type'):
                    new_rettype = TensorType(f.ret_type.shape, dtype=obj.var_type)
                else:
                    new_rettype = None
                new_body = self.visit(f.body)
                new_f = relay.function.Function(params=new_params, body=new_body, ret_type=new_rettype)
                return new_f

            def visit_call(self, call):
                new_fn = self.visit(call.op)
                new_args = [self.visit(arg) for arg in call.args]
                # if hasattr(call.attrs, 'out_dtype') and len(call.attrs.out_dtype) > 0:
                #     call.attrs.out_dtype = obj.var_type
                #     call.attrs.__setattr__('out_dtype', 'int8')
                # # print(call.attrs)
                # return relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)
                new_types = []
                for ta in call.type_args:
                #    if hasattr(ta, 'shape'):
                    nt = TensorType(ta.shape, dtype=obj.var_type)
                    new_types.append(nt)
                if hasattr(call.attrs, 'out_dtype') and len(call.attrs.out_dtype) > 0:
                    # my_type = type(call.attrs)
                    # no = my_type()
                    # ATTENTION: this doesn't work, since copying the handle means (apparently) to point to the very
                    #  same read-only C object.
                    # no.handle = call.attrs.handle
                    # # no.out_dtype = obj.var_type
                    # for field in call.attrs.list_field_info():
                    #     if isinstance(field.name, str) and 'dtype' in field.name:
                    #         # no[field.name] = obj.var_type
                    #         setattr(no, field.name, obj.var_type)
                    #     else:
                    #         # no[field.name] = field
                    #         setattr(no, field.name, call.attrs[field.name])
                    attrs_relay_type = 'relay.attrs.' + type(call.attrs).__name__
                    n_fields = dict()
                    for field in call.attrs.list_field_info():
                        if isinstance(field.name, str) and 'dtype' in field.name:
                            # ATTENTION: the 'str()' is important! Because field.name is of type String,
                            #  the packing (ffi) will fail otherwise!!
                            # The Error is " Check failed: args.type_codes[i] == kTVMStr (8 vs. 11) :" and arises from
                            # TVM FFI argument type encoding:
                            # https://tvm.apache.org/docs/reference/api/doxygen/c__runtime__api_8h.html#a190e81769e805cca153514137a66e793af2b95de1a09ed39055edfe8ef5ea484d
                            n_fields[str(field.name)] = obj.var_type
                        else:
                            # ATTENTION: the 'str()' is important! Because field.name is of type String,
                            #  the packing (ffi) will fail otherwise!!
                            # (see above)
                            n_fields[str(field.name)] = call.attrs[field.name]
                    # fields = {'units': None}
                    new_attrs = tvm.ir.make_node(attrs_relay_type, **n_fields)
                else:
                    new_attrs = call.attrs
                new_call = relay.expr.Call(new_fn, new_args, type_args=new_types, attrs=new_attrs, span=call.span)
                return new_call

            # def visit_op(self, o):
            #     return o

        return GlobalCastDtypes().visit(func)

