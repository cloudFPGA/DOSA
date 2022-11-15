import torch.nn as nn
import brevitas.nn as qnn
from src.models.quantized import QuantModule
from src.utils import Reshape


class QResidualBlock(QuantModule):
    def __init__(self, in_channels, out_channels, stride=1):
        super(QResidualBlock, self).__init__()
        self.downsample = False

        # first convolutional
        self._append(qnn.QuantConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
                                     weight_quant=None))  # 0
        self._append(nn.BatchNorm2d(out_channels)),  # 1
        self._append(qnn.QuantReLU(act_quant=None))  # 2

        # second convolutional
        self._append(qnn.QuantConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                                     weight_quant=None))  # 3
        self._append(nn.BatchNorm2d(out_channels))  # 4

        # downsample
        if stride != 1 or in_channels != out_channels:
            self.downsample = True
            self._append(qnn.QuantConv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False,
                                         weight_quant=None))  # 5
            self._append(nn.BatchNorm2d(out_channels))  # 6
        else:
            self._append(qnn.QuantIdentity(act_quant=None))  # 5

        # relu
        self._append(qnn.QuantReLU(act_quant=None))  # 6 or 7

    def forward(self, x):
        out = x
        x_downsampled = None
        for i, module in enumerate(self.features):
            if self.downsample and i == 5:
                x_downsampled = module(x)
            elif self.downsample and i == 6:
                x_downsampled = module(x_downsampled)
                out += x_downsampled
            elif not self.downsample and i == 5:
                out += module(x)
            else:
                out = module(out)
        return out


class QResNet(QuantModule):
    """
    Base class for quantized ResNet, per default not quantized and therefore acts as a wrapper for the full precision
    model
    """

    def __init__(self, block, layers, num_classes=10):
        super(QResNet, self).__init__()
        self.inplanes = 64

        # first layer
        self._append(qnn.QuantConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False,
                                     weight_quant=None))
        self._append(nn.BatchNorm2d(64))
        self._append(qnn.QuantReLU(act_quant=None))

        # block layers
        self._make_layer(block, 64, layers[0], stride=1)
        self._make_layer(block, 128, layers[1], stride=2)
        self._make_layer(block, 256, layers[2], stride=2)
        self._make_layer(block, 512, layers[3], stride=2)

        # last layer
        self._append(nn.AvgPool2d(4))
        self._append(Reshape(lambda x: (x.shape[0], -1)))
        self._append(qnn.QuantLinear(512, num_classes, bias=True,
                                     weight_quant=None))

    def _make_layer(self, block, planes, blocks, stride=1):
        strides = [stride] + [1]*(blocks-1)
        for stride in strides:
            self._append(block(self.inplanes, planes, stride))
            self.inplanes = planes

    def forward(self, x):
        for module in self.features:
            x = module(x)
        return x


class QResNet18(QResNet):
    def __init__(self):
        super(QResNet18, self).__init__(QResidualBlock, [2, 2, 2, 2])


class QResNet34(QResNet):
    def __init__(self):
        super(QResNet34, self).__init__(QResidualBlock, [3, 4, 6, 3])
