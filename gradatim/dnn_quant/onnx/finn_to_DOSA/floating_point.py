import numpy as np
from onnx import helper as oh
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
    return model.get_tensor_datatype(node.input[1]) == DataType["FLOAT32"]


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

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for n in graph.node:
            if involves_float_operation(model, n):
                remove_node_and_rewire(model, n)
                graph_modified = True
                break
        return model, graph_modified

