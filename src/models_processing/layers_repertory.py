import torch.nn as nn
import brevitas.nn as qnn

quantizable_translation_weight = {
    nn.Conv1d.__name__: qnn.QuantConv1d,
    nn.Conv2d.__name__: qnn.QuantConv2d,
    nn.Linear.__name__: qnn.QuantLinear
}

quantizable_translation_activations = {
    nn.Identity.__name__: qnn.QuantIdentity,
    nn.ReLU.__name__: qnn.QuantReLU,
    nn.Sigmoid.__name__: qnn.QuantSigmoid,
}

quantizable_translation_all = {
    **quantizable_translation_weight,
    **quantizable_translation_activations
}

weight_layers_all = [
    nn.Conv1d.__name__,
    nn.Conv2d.__name__,
    nn.Linear.__name__,
    qnn.QuantConv1d.__name__,
    qnn.QuantConv2d.__name__,
    qnn.QuantLinear.__name__
]
