'''VGG for CIFAR10.
(c) YANG, Wei
'''
import torch.nn as nn
import math
import torch.nn.init as init
from modules import *


__all__ = ['vgg_small_1w1a']


class VGG(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.nonlinear0 = nn.Hardtanh(inplace=True)
        self.conv1 = BinarizeConv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.conv2 = BinarizeConv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.conv3 = BinarizeConv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.nonlinear3 = nn.Hardtanh(inplace=True)
        self.conv4 = BinarizeConv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.nonlinear4 = nn.Hardtanh(inplace=True)
        self.conv5 = BinarizeConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.nonlinear5 = nn.Hardtanh(inplace=True)
        self.fc = nn.Linear(512*4*4, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BinarizeConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.nonlinear0(x)
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear2(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear4(x)
        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.nonlinear5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def vgg_small_1w1a(**kwargs):
    model = VGG(**kwargs)
    return model


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda name: 'conv' in name or 'fc' in name, [name[0] for name in list(net.named_modules())]))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('vgg'):
            print(net_name)
            test(globals()[net_name]())
            print()