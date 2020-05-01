import torch
import torch.nn as nn
import torch.nn.functional as F


# The block class contains a set of conv/norm/residual sequence that makes up the building block of the repeating residual
# layers of ResNet
class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()

        # expansion is a way to indicate whether of not if the number of output channels at the end of a layer has changed or not
        # triggering the identity_downsample in order to change the number of input channel for the next layer
        self.expansion = 4

        # conv layer of a ResNet block
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0)
        self.BN1 = nn.BatchNorm2d(intermediate_channels)

        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1)
        self.BN2 = nn.BatchNorm2d(intermediate_channels)

        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels * self.expansion, kernel_size=1, stride=1,
                               padding=0)
        self.BN3 = nn.BatchNorm2d(intermediate_channels * self.expansion)

        self.relu = nn.ReLU()

        # updates the identity downsample at end of each block
        self.identity_downsample = identity_downsample

    def forward(self, x):
        # sets the identity as x
        identity = x

        x = self.conv1(x)
        x = self.BN1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.BN2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.BN3(x)
        x = self.relu(x)

        # Detects if layers has changed and changes the identity to the same size as the current output so it can be added
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        # adds the identity(residual) to the current output at the end of each block
        x = x + identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):

    # block: block class
    # layers: list of how many times the block will be re-used in each layer of the net
    #        Ex: ResNet 50: layers = [3, 4, 6, 3] layer1 will use block 3 times
    #                                             layer2 will use block 4 times...
    # image_channe: 1 or 3 (grayscale or RGB)
    # num_classes: how many identification classes
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()

        # The initial convolution layer before the data gets sent through to the ResNet blocks
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.BN1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Resnet layers that call a _make_layer function that strings together resnet blocks to make the layer
        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=2)

        # output = out_channels * expansion factor(4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        # the inital layer BEFORE using the residual model blocks
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # the layers that will call the _make_layer function that will be made up of multiple
        # blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_block, intermediate_channels, stride):
        # block: block class
        # num_block: how many blocks the layer will be made out of (layers.item())
        # intermediate_channels: number of output after the layer (as well as what all the blocks will output in that layer)
        # stride: each layer that will be used in the block will have same stride
        identity_downsample = None
        layers = []

        # detects when it is time to update the identity in order to add as the residual to the current output
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            # a short conv layer that will increase the output size to match current output
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, intermediate_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(intermediate_channels * 4))
        # This is the first block of the layer
        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * 4

        # loops throught the rest of the blocks required for the layer
        # Example: if 1st layer uses 3 blocks, the first is already implemented above
        #          therefore loops 3-1 more times
        for i in range(num_block - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

#------------functions that can be implement the different ResNet models depending on input parameters-----------
def ResNet50(img_channel):
    num_classes = 7
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)

def ResNet101(img_channel):
    num_classes = 1000
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)

def ResNet152(img_channel):
    num_classes = 1000
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)
