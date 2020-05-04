import torch
import torch.nn as nn
import torch.nn.functional as F

# ResNet30 and ResNet18 implementation is the same overall structure to the other versions
# But it only uses the basic_block featuring 2 convolution layers instead of 3 like the bottleNeck block

class basic_block(nn.Module):
    blk_expand = 4

    def __init__(self, in_channels, map_back_channels, change_residual_shape=None, stride=1):
        super(basic_block, self).__init__()

        # Only 2 convolution layers are needed and no expansion is needed
        self.conv1 = nn.Conv2d(in_channels, map_back_channels, kernel_size=3, stride=stride, padding=1)
        self.BN1 = nn.BatchNorm2d(map_back_channels)

        self.conv2 = nn.Conv2d(map_back_channels, map_back_channels, kernel_size=3, stride=1, padding=1)
        self.BN2 = nn.BatchNorm2d(map_back_channels)

        self.change_residual_shape = change_residual_shape

    def forward(self, x):
        Residual = x

        x = self.conv1(x)
        x = F.relu(self.BN1(x))
        x = self.conv2(x)
        x = F.relu(self.BN2(x))

        if self.change_residual_shape is not None:
            Residual = self.change_residual_shape(Residual)

        x = x + Residual

        x = F.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, basic_block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.map_back_current = 64

        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.BN1 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.resLayer1 = self.make_layer(basic_block, layers[0], map_back_channels=64, stride=1)
        self.resLayer2 = self.make_layer(basic_block, layers[1], map_back_channels=128)
        self.resLayer3 = self.make_layer(basic_block, layers[2], map_back_channels=256)
        self.resLayer4 = self.make_layer(basic_block, layers[3], map_back_channels=512)

        # No expansion therefore the output will have same number of outputs as the the map_back_channels value
        self.fc = nn.Linear(512, num_classes)
        self.do = nn.Dropout(p=0.5)

    def make_layer(self, basic_block, num_blocks, map_back_channels, stride=2):

        change_residual_shape = None
        layer_blocks = []

        self.resConv = nn.Conv2d(self.in_channels, map_back_channels, kernel_size=1, stride=stride)
        self.resBatch = nn.BatchNorm2d(map_back_channels)

        # only conditions are if the layers has changed which can be detected by storing a copy of map_back_channels\
        # to compare with
        if stride != 1 or self.map_back_current != map_back_channels:
            self.map_back_current = map_back_channels
            change_residual_shape = nn.Sequential(self.resConv,
                                                  self.resBatch)

        layer_blocks.append(basic_block(self.in_channels, map_back_channels, change_residual_shape, stride))

        self.in_channels = map_back_channels

        for i in range(num_blocks - 1):
            layer_blocks.append(basic_block(self.in_channels, map_back_channels))

        return nn.Sequential(*layer_blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.BN1(x))
        x = self.maxpool(x)

        x = self.resLayer1(x)
        x = self.resLayer2(x)
        x = self.resLayer3(x)
        x = self.resLayer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.do(x)
        x = self.fc(x)


        return x

#------------------------------------- ResNet selection -------------------------------------------#
def ResNet30(img_channel):
    num_classes = 7
    return ResNet(basic_block, [3, 4, 6, 3], img_channel, num_classes)

def ResNet18(img_channel):
    num_classes = 7
    return ResNet(basic_block, [2, 2, 2, 2], img_channel, num_classes)