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
import torch.nn as nn
import brevitas.nn as qnn

brevitas_translation_weight_layers = {
    nn.Linear.__name__: qnn.QuantLinear,
    nn.Conv1d.__name__: qnn.QuantConv1d,
    nn.Conv2d.__name__: qnn.QuantConv2d,
    nn.ConvTranspose1d.__name__: qnn.QuantConvTranspose1d,
    nn.ConvTranspose2d.__name__: qnn.QuantConvTranspose2d,
}

brevitas_translation_stateful_layers = {
    **brevitas_translation_weight_layers,
    nn.BatchNorm1d.__name__: nn.BatchNorm1d,
    nn.BatchNorm2d.__name__: nn.BatchNorm2d,
}

brevitas_translation_activations = {
    nn.Identity.__name__: qnn.QuantIdentity,
    nn.ReLU.__name__: qnn.QuantReLU,
    nn.Sigmoid.__name__: qnn.QuantSigmoid,
    nn.Tanh.__name__: qnn.QuantTanh,
    nn.Hardtanh.__name__: qnn.QuantHardTanh,
}

brevitas_pooling = {
    nn.AvgPool2d.__name__: qnn.QuantAvgPool2d,
    nn.AdaptiveAvgPool2d.__name__: qnn.QuantAdaptiveAvgPool2d,
    nn.MaxPool1d.__name__: qnn.QuantMaxPool1d,
    nn.MaxPool2d.__name__: qnn.QuantMaxPool2d,
}

brevitas_upsampling = {
    nn.Upsample.__name__: qnn.QuantUpsample,
    nn.UpsamplingBilinear2d.__name__: qnn.QuantUpsamplingBilinear2d,
    nn.UpsamplingNearest2d.__name__: qnn.QuantUpsamplingNearest2d,
}

brevitas_translation_other = {
    nn.Dropout.__name__: qnn.QuantDropout,
}

brevitas_translation_all = {
    **brevitas_translation_weight_layers,
    **brevitas_translation_activations,
    **brevitas_pooling,
    **brevitas_upsampling,
    **brevitas_translation_other,
}

weight_layers_all = list(brevitas_translation_weight_layers.keys()) + \
                    list(map(lambda x: x.__name__, list(brevitas_translation_weight_layers.values())))


nn_modules_all = list(brevitas_translation_all.keys()) + \
                 list(map(lambda x: x.__name__, list(brevitas_translation_all.values()))) + \
                 [qnn.QuantCat.__name__, qnn.QuantEltwiseAdd.__name__, qnn.QuantScaleBias.__name__]
