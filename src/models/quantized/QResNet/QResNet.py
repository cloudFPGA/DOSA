import torch.nn as nn
import brevitas.nn as qnn
from src.models.quantized import QuantModule
from src.utils import Reshape


class QResidualBlock(QuantModule):
    def __init__(self, in_channels, out_channels, stride=1,
                 act_quant=None,
                 weight_quant=None,
                 bias_quant=None,
                 bit_width=None):
        super(QResidualBlock, self).__init__(num_act=4, num_weighted=3, num_biased=0)

        a_quant, w_quant, b_quant, bit_width, return_qt, do_quantization = \
            self._process_quant_methods(act_quant, weight_quant, bias_quant, bit_width)

        self.downsample = False
        self.forward_step_index = 0
        self.forward_step_input_x = None
        self.forward_step_output6 = None

        # first convolutional
        self._append(qnn.QuantConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
                                     return_quant_tensor=False, weight_quant=w_quant[0]))  # 0
        self._append(nn.BatchNorm2d(out_channels)),  # 1
        self._append(qnn.QuantIdentity(act_quant=a_quant[1], return_quant_tensor=return_qt))  # 2
        if do_quantization:
            self._append(qnn.QuantReLU(return_quant_tensor=return_qt, bit_width=bit_width))  # 3
        else:
            self._append(qnn.QuantReLU(act_quant=None))  # 3

        # second convolutional
        self._append(qnn.QuantConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                                     return_quant_tensor=False, weight_quant=w_quant[1]))  # 4
        self._append(nn.BatchNorm2d(out_channels))  # 5
        qidd_add = qnn.QuantIdentity(act_quant=a_quant[2], return_quant_tensor=return_qt)
        self._append(qidd_add)  # 6

        # downsample
        if stride != 1 or in_channels != out_channels:
            self.downsample = True
            self._append(qnn.QuantIdentity(act_quant=a_quant[3], return_quant_tensor=return_qt))  # 7
            self._append(qnn.QuantConv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False,
                                         return_quant_tensor=False, weight_quant=w_quant[2]))  # 8
            self._append(nn.BatchNorm2d(out_channels))  # 9
            self._append(qidd_add)  # 10
        else:
            self._append(qidd_add)  # 7

        # relu
        if do_quantization:
            self._append(qnn.QuantReLU(return_quant_tensor=return_qt, bit_width=bit_width))  # 8 or 11
        else:
            self._append(qnn.QuantReLU(act_quant=None))  # 8 or 11

    def forward(self, x):
        out = x
        x_downsampled = None
        for i, module in enumerate(self.features):
            if self.downsample and i == 7:
                x_downsampled = module(x)
            elif self.downsample and (i == 8 or i == 9):
                x_downsampled = module(x_downsampled)
            elif self.downsample and i == 10:
                x_downsampled = module(x_downsampled)
                out += x_downsampled
            elif not self.downsample and i == 7:
                out += module(x)
            else:
                out = module(out)
        return out

    def forward_step(self, x):
        if self.forward_step_index >= len(self.features):
            self.forward_step_index = 0
            return None, None, None

        module = self.features[self.forward_step_index]

        if self.forward_step_index == 0:
            self.forward_step_input_x = x
            x_in = x
            x_out = module(x_in)

        elif self.forward_step_index == 6:
            x_in = x
            x_out = module(x_in)
            self.forward_step_output6 = x_out

        elif self.forward_step_index == 7:
            x_in = self.forward_step_input_x
            x_out = module(x_in) if self.downsample else x + module(x_in)

        elif self.downsample and self.forward_step_index == 10:
            x_in = x
            x_out = self.forward_step_output6 + module(x_in)

        else:
            x_in = x
            x_out = module(x_in)

        self.forward_step_index += 1
        return x_in, module, x_out


class QResNet(QuantModule):
    """
    Base class for quantized QResNet, per default not quantized and therefore acts as a wrapper for the full precision
    model
    """
    def __init__(self, block, layers, num_classes=10,
                 act_quant=None,
                 weight_quant=None,
                 bias_quant=None,
                 bit_width=None):

        super(QResNet, self).__init__(num_act=7, num_weighted=6, num_biased=1)
        self.forward_step_index = 0

        self.inplanes = 64

        a_quant, w_quant, b_quant, bit_width, return_qt, do_quantization = \
            self._process_quant_methods(act_quant, weight_quant, bias_quant, bit_width)

        # first layer
        self._append(qnn.QuantIdentity(act_quant=a_quant[0], return_quant_tensor=return_qt))
        self._append(qnn.QuantConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, return_quant_tensor=False,
                                     weight_quant=w_quant[0]))
        self._append(nn.BatchNorm2d(64))
        self._append(qnn.QuantIdentity(act_quant=a_quant[1], return_quant_tensor=return_qt))
        if do_quantization:
            self._append(qnn.QuantReLU(return_quant_tensor=return_qt, bit_width=bit_width))
        else:
            self._append(qnn.QuantReLU(act_quant=None))

        # block layers
        self._make_layer(block, 64, layers[0], stride=1, act_quant=a_quant[2], weight_quant=w_quant[1],
                         bit_width=bit_width)
        self._make_layer(block, 128, layers[1], stride=2, act_quant=a_quant[3], weight_quant=w_quant[2],
                         bit_width=bit_width)
        self._make_layer(block, 256, layers[2], stride=2, act_quant=a_quant[4], weight_quant=w_quant[3],
                         bit_width=bit_width)
        self._make_layer(block, 512, layers[3], stride=2, act_quant=a_quant[5], weight_quant=w_quant[4],
                         bit_width=bit_width)

        # last layer
        if do_quantization:
            self._append(qnn.QuantAvgPool2d(4, bit_width=bit_width))
        else:
            self._append(nn.AvgPool2d(4))
        self._append(Reshape(lambda x: (x.shape[0], -1)))
        self._append(qnn.QuantIdentity(act_quant=a_quant[6], return_quant_tensor=return_qt))
        self._append(qnn.QuantLinear(512, num_classes, bias=True, bias_quant=b_quant[0], weight_quant=w_quant[5]))

    def _make_layer(self, block, planes, blocks, stride=1,
                    act_quant=None,
                    weight_quant=None,
                    bias_quant=None,
                    bit_width=None):
        strides = [stride] + [1] * (blocks - 1)
        for stride in strides:
            self._append(block(self.inplanes, planes, stride,
                               act_quant=act_quant, weight_quant=weight_quant, bias_quant=bias_quant,
                               bit_width=bit_width))
            self.inplanes = planes

    def forward(self, x):
        for module in self.features:
            x = module(x)
        return x

    def forward_step(self, x):
        if self.forward_step_index >= len(self.features):
            self.forward_step_index = 0
            return None, None, None

        module = self.features[self.forward_step_index]
        out = module(x)
        self.forward_step_index += 1
        return x, module, out


class QResNet18(QResNet):
    def __init__(self, act_quant=None, weight_quant=None, bias_quant=None, bit_width=None):
        super(QResNet18, self).__init__(QResidualBlock, [2, 2, 2, 2],
                                        act_quant=act_quant,
                                        weight_quant=weight_quant,
                                        bias_quant=bias_quant,
                                        bit_width=bit_width)


class QResNet34(QResNet):
    def __init__(self, act_quant=None, weight_quant=None, bias_quant=None, bit_width=None):
        super(QResNet34, self).__init__(QResidualBlock, [3, 4, 6, 3],
                                        act_quant=act_quant,
                                        weight_quant=weight_quant,
                                        bias_quant=bias_quant,
                                        bit_width=bit_width)
