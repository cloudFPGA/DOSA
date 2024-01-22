import numpy as np
from onnx import helper as oh
from onnx import numpy_helper
from qonnx.core.datatype import DataType
from qonnx.transformation.base import Transformation
from qonnx.transformation.remove import remove_node_and_rewire


def is_float_add(model, node):
    assert node.op_type == "Add"
    param = model.get_initializer(node.input[1])
    return (param != param.round()).any()


def is_float_mul(model, node):
    assert node.op_type == "Mul"
    param = model.get_initializer(node.input[1])
    return (param != param.round()).any()


def is_float_multithreshold(model, node):
    assert node.op_type == "MultiThreshold"
    # return model.get_tensor_datatype(node.input[1]) == DataType["FLOAT32"]
    # to catch incorrect labelled brevitas nodes: all input nodes must be float
    still_float = True
    for i in range(len(node.input)):
        if model.get_tensor_datatype(node.input[i]) != DataType["FLOAT32"]:
            still_float = False
    return still_float


def involves_float_operation(model, node):
    if node.op_type == "MultiThreshold" and is_float_multithreshold(model, node):
        return True
    if node.op_type == "Add" and is_float_add(model, node):
        return True
    if node.op_type == "Mul" and is_float_mul(model, node):
        return True
    return False


class RemoveFloatPointNodes(Transformation):
    """Remove nodes involving full precision input data from the model."""

    def __init__(self, returnlist_removed_nodes=None):
        super().__init__()
        self.returnlist_removed_nodes = returnlist_removed_nodes

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for n in graph.node:
            if involves_float_operation(model, n):
                if self.returnlist_removed_nodes is not None:
                    initializers = []
                    numpy_data = []
                    for inits_id in range(len(graph.initializer)):
                        init = graph.initializer[inits_id]
                        if init.name in n.input:
                            initializers.append(init)
                            numpy_data.append(numpy_helper.to_array(init))
                    ne = {'node': n, 'initializers': initializers, 'numpy_data': numpy_data}
                    self.returnlist_removed_nodes.append(ne)
                remove_node_and_rewire(model, n)
                graph_modified = True
                break
        return model, graph_modified

