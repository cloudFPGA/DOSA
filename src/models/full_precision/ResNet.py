from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        print('    ', x.shape, ' ', 'Conv2d 1')
        out = self.conv1(x)
        print(self.conv1[0].out_channels, ' ', self.conv1[0].stride)
        print('    ', x.shape, ' ', 'Conv2d 2')
        out = self.conv2(out)
        print('    ', x.shape, ' ', 'downsample')
        out += self.downsample(x)
        print('    ', x.shape, ' ', 'relu')
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for stride in strides:
            print(self.inplanes, ' ', planes)
            layers.append(block(self.inplanes, planes, stride))   
            self.inplanes = planes
        print()
        return nn.Sequential(*layers)

    def forward(self, x):
        print(x.shape, ' ', 'Conv2d')
        x = self.conv1(x)
        print(x.shape, ' ', 'ResBlock1,2')
        x = self.layer0(x)
        print(x.shape, ' ', 'ResBlock3, 4')
        x = self.layer1(x)
        print(x.shape, ' ', 'ResBlock5, 6')
        x = self.layer2(x)
        print(x.shape, ' ', 'ResBlock7, 8')
        x = self.layer3(x)

        print(x.shape, ' ', 'avgpool')
        x = F.avg_pool2d(x, 4)
        print(x.shape, ' ', 'reshape')
        x = x.view(x.size(0), -1)
        print(x.shape, ' ', 'linear')
        x = self.fc(x)
        
        return x

    
def ResNet18():
    return ResNet(ResidualBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(ResidualBlock, [3, 4, 6, 3])
