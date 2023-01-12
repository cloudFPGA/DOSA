import numpy as np
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.transformation.base import Transformation


class RemoveThresModuleInfo(Transformation):
    """ Remove "module" information of MultiThreshold nodes."""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for n in graph.node:
            if n.op_type == "MultiThreshold":
                # remove "module
                if n.domain:
                    n.domain = ''
                    graph_modified = True
        return model, graph_modified


class ThresMissingOutBiasToZero(Transformation):
    """Set 'out_bias' to zero when the attribute is missing"""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for n in graph.node:
            if n.op_type == "MultiThreshold":
                # check 'out_bias' exists, if not create one set to zero
                if 'out_bias' not in [a.name for a in n.attribute]:
                    out_bias = oh.make_attribute('out_bias', 0.0)
                    n.attribute.append(out_bias)
                    graph_modified = True

        return model, graph_modified


class ConvertThresToAdd(Transformation):
    """Convert MultiThreshold nodes to Add."""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
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
                graph_modified = True
        return model, graph_modified
