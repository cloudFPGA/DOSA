#  /*******************************************************************************
#   * Copyright 2022 -- 2024 IBM Corporation
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
from onnx import helper
import warnings

custom_opsets = {
    'finn.custom_op.general': helper.make_operatorsetid('finn.custom_op.general', 1),
    'qonnx.custom_op.general': helper.make_operatorsetid('qonnx.custom_op.general', 1)
}


def all_nodes_opsets(model):
    opsets = []
    for node in model.graph.node:
        if node.domain not in opsets:
            opsets.append(node.domain)
    return opsets

"""
SOP: fix_missing_opset was meant to resolve and make this warning disappear. The problem has nothing to do with TVM though, 
it was the onnx model that was not consistent with itself.  The individual nodes in the model had opset attributes that 
were not defined in the onnx model metadata  (ie. model.opset_import vs. node.domain).
"""

def fix_missing_opsets(model):
    model_opsets = [o.domain for o in model.model.opset_import]
    nodes_opsets = all_nodes_opsets(model)
    missing_opsets = [op for op in nodes_opsets if op not in model_opsets]
    unrecognized_opsets = []
    for op in missing_opsets:
        op_object = custom_opsets.get(op, None)
        if op_object:
            model.model.opset_import.append(op_object)
        else:
            unrecognized_opsets.append(op)
    if unrecognized_opsets:
        msg = "The following opsets are not recognized: "
        msg += ", ".join(unrecognized_opsets)
        warnings.warn(msg)
    return model


