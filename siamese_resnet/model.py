from PIL import Image
from numpy.lib.arraypad import pad
import torch
import torch.nn as nn
import config as c


class ResBlock(nn.Module):
    def __init__(self, c_in, c_mid, stride=1, identity_fn=None):
        super().__init__()
        self.identitiy_fn = identity_fn
        self.conv1 = nn.Conv2d(c_in, c_mid, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.conv2 = nn.Conv2d(c_mid, c_mid, kernel_size=3, stride=stride, 
                               padding=1)
        self.bn2 = nn.BatchNorm2d(c_mid)
        self.conv3 = nn.Conv2d(c_mid, c_mid * 4, 
                               kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(c_mid * 4)
        self.elu = nn.ELU()

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identitiy_fn is not None:
            identity = self.identitiy_fn(identity)

        x += identity
        x = self.elu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, layers, img_channels, n_output_features):
        super().__init__()
        self.c_in = 64
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, 
                               padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.elu = nn.ELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_layer1 = self._make_res_layer(layers[0], 64, stride=1)
        self.res_layer2 = self._make_res_layer(layers[1], 128, stride=2)
        self.res_layer3 = self._make_res_layer(layers[2], 256, stride=2)
        self.res_layer4 = self._make_res_layer(layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear1 = nn.Linear(512 * 4, n_output_features)
        self.linear2 = nn.Linear(n_output_features, 1)

    def _make_res_layer(self, n_blocks, c_mid, stride):
        identity_fn = None
        layers = []

        if stride != 1 or self.c_in != c_mid * 4:
            identity_fn = nn.Sequential(nn.Conv2d(self.c_in, c_mid * 4,
                                                  kernel_size=1, stride=stride,
                                                  padding=0),
                                        nn.BatchNorm2d(c_mid * 4))
        layers.append(ResBlock(self.c_in, c_mid, stride, identity_fn))
        self.c_in = c_mid * 4

        for i in range(n_blocks - 1):
            layers.append(ResBlock(self.c_in, c_mid))
        
        return nn.Sequential(*layers)

    def forward(self, x_1, x_2):
        x_1 = self.conv1(x_1)
        x_1 = self.bn(x_1)
        x_1 = self.elu(x_1)
        x_1 = self.maxpool(x_1)
        x_1 = self.res_layer1(x_1)
        x_1 = self.res_layer2(x_1)
        x_1 = self.res_layer3(x_1)
        x_1 = self.res_layer4(x_1)
        x_1 = self.avgpool(x_1)
        x_1 = x_1.reshape(x_1.shape[0], -1)
        x_1 = self.linear1(x_1)

        x_2 = self.conv1(x_2)
        x_2 = self.bn(x_2)
        x_2 = self.elu(x_2)
        x_2 = self.maxpool(x_2)
        x_2 = self.res_layer1(x_2)
        x_2 = self.res_layer2(x_2)
        x_2 = self.res_layer3(x_2)
        x_2 = self.res_layer4(x_2)
        x_2 = self.avgpool(x_2)
        x_2 = x_2.reshape(x_2.shape[0], -1)
        x_2 = self.linear1(x_2)

        x_dist = torch.abs((x_1 - x_2))
        x_dist = self.linear2(x_dist)

        return x_dist.squeeze()

def ResNet50(img_channels, n_output_features):
    return ResNet([3, 4, 6, 3], img_channels, n_output_features)

def ResNet101(img_channels, n_output_features):
    return ResNet([3, 4, 23, 3], img_channels, n_output_features)

def ResNet152(img_channels, n_output_features):
    return ResNet([3, 8, 36, 3], img_channels, n_output_features)
