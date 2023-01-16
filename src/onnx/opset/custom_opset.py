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


