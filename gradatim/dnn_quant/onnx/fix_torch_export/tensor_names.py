from onnx import helper


def check_quantization_annotation(graph, initializer_name, new_name):
    seen_first = False
    for quant_annot_index, quant_annot in enumerate(graph.quantization_annotation):
        if quant_annot.tensor_name == initializer_name:
            if not seen_first:
                seen_first = True
            else:
                quant_annot.tensor_name = new_name
                break


def rename_initializer(node, graph, initializer_index, initializer_name, new_name):
    initializer = [i for i in graph.initializer if i.name == initializer_name]
    assert len(initializer) == 1, "ERROR: multiple initializers with the same name."
    initializer = initializer[0]

    new_initializer = helper.make_tensor(new_name, initializer.data_type, initializer.dims,
                                         initializer.raw_data, raw=True)
    node.input[initializer_index] = new_name
    graph.initializer.append(new_initializer)


def fix_shared_initializers(model):
    """
    Here we fix a problem appearing in Pytorch version 1.13, where different nodes in the exported onnx share the
    same initializer object, whereas there should be two distincts ones.
    """
    graph = model.graph
    graph_initializers = [i.name for i in graph.initializer]
    seen_initializer = []

    for node_index, n in enumerate(graph.node):
        for initializer_index, initializer_name in enumerate(n.input):
            if initializer_name in graph_initializers:
                # first occurrence of initializer name
                if initializer_name not in seen_initializer:
                    seen_initializer.append(initializer_name)

                # multiple occurrences of initializer names: two different nodes share the same initializer
                else:
                    new_name = initializer_name + str(node_index)
                    rename_initializer(n, graph, initializer_index, initializer_name, new_name)
                    check_quantization_annotation(graph, initializer_name, new_name)

    model.check_all_tensor_shapes_specified(fix_missing_init_shape=True)
    return model



