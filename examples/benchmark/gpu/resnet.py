from torch import nn
import torch.nn.functional as F
import pytorch_quantization.nn as quant_nn


class TensorrtResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(TensorrtResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            quant_nn.QuantConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            quant_nn.QuantConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()
        self.out_channels = out_channels

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                quant_nn.QuantConv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.downsample(x)
        out = self.relu(out)
        return out


class TensorrtResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(TensorrtResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Sequential(
            quant_nn.QuantConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = quant_nn.QuantLinear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def TensorrtResNet18():
    return TensorrtResNet(TensorrtResidualBlock, [2, 2, 2, 2])
