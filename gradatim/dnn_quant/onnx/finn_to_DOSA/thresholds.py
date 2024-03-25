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
import numpy as np
from onnx import helper as oh
from qonnx.transformation.base import Transformation


class RemoveThresModuleInfo(Transformation):
    """Remove "module" information of MultiThreshold nodes."""

    def apply(self, model):
        graph = model.graph
        for n in graph.node:
            if n.op_type == "MultiThreshold":
                # remove "module
                if n.domain:
                    n.domain = ''
        return model, False


class ThresMissingOutBiasToZero(Transformation):
    """Set 'out_bias' to zero when the attribute is missing"""

    def apply(self, model):
        graph = model.graph
        for n in graph.node:
            if n.op_type == "MultiThreshold":
                # check 'out_bias' exists, if not create one set to zero
                if 'out_bias' not in [a.name for a in n.attribute]:
                    out_bias = oh.make_attribute('out_bias', 0.0)
                    n.attribute.append(out_bias)

        return model, False


class ConvertThresToAdd(Transformation):
    """Convert MultiThreshold nodes to Add."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            if n.op_type == "MultiThreshold":
                # create Add
                thres_input_shape = model.get_tensor_shape(n.input[0])
                add_param = np.zeros(thres_input_shape, dtype='float32')
                add_param_name = model.make_new_valueinfo_name()
                model.set_initializer(add_param_name, add_param)
                # create new node
                add_node = oh.make_node(
                    "Add",
                    [n.input[0], add_param_name],
                    n.output,
                )
                # remove old node, add new node to graph at correct position
                graph.node.insert(node_ind, add_node)
                graph.node.remove(n)
        return model, False
