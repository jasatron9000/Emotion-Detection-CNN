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

        # If a change_residual_shape was given as a input then it means the map_back_channels
        # has been changed(going into the next ResNet layer)
        #
        # Therefore the Sequential for changing the residual to match the current output(x)
        # must be updated
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
    def __init__(self, basic_block, layers, num_classes):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.map_back_current = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
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

        # Define convolution and batch normalisation layer for changing the residual shape
        self.resConv = nn.Conv2d(self.in_channels, map_back_channels, kernel_size=1, stride=stride)
        self.resBatch = nn.BatchNorm2d(map_back_channels)

        # only conditions are if the layers has changed which can be detected by storing a copy of map_back_channels\
        # to compare with
        if stride != 1 or self.map_back_current != map_back_channels:
            self.map_back_current = map_back_channels
            change_residual_shape = nn.Sequential(self.resConv,
                                                  self.resBatch)

        # A initial block will be appended to the the list first as that block will have the stride value that is
        # giving when making the layers, which is in charge of decreasing the the dimension on the feature maps(by half)
        layer_blocks.append(basic_block(self.in_channels, map_back_channels, change_residual_shape, stride))

        # The in_channel value must also update at the start of every layer as it should remain to be the map_back_channels
        self.in_channels = map_back_channels

        # A for loop is used (in range num_blocks - 1 because already appended previously) to
        # add on the rest of the bottleNeck_blocks for the layer(stride is already hard coded to 1 in the bottleNeck_block class)
        for i in range(num_blocks - 1):
            layer_blocks.append(basic_block(self.in_channels, map_back_channels))

        # Finally returns a pytorch Sequential that strings together all the bottleNeck blocks together
        return nn.Sequential(*layer_blocks)

    def forward(self, x):

        # Going through inital layer BEFORE using the ResNet layers that are made up of basic_block blocks
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = F.relu(self.BN1(x))
        print(x.shape)
        x = self.maxpool(x)
        print(x.shape)

        # Going through the Resnet layers
        x = self.resLayer1(x)
        print(x.shape)
        x = self.resLayer2(x)
        print(x.shape)
        x = self.resLayer3(x)
        print(x.shape)
        x = self.resLayer4(x)

        # finally to a fully connected layer with 512 * blk_expansion connections
        print(x.shape)
        x = self.avgpool(x)
        print(x.shape)
        x = x.reshape(x.shape[0], -1)
        x = self.do(x)
        print(x.shape)
        x = self.fc(x)


        return x

#------------------------------------- ResNet selection -------------------------------------------#
def ResNet34():
    num_classes = 7
    return ResNet(basic_block, [3, 4, 6, 3], num_classes)

def ResNet18():
    num_classes = 7
    return ResNet(basic_block, [2, 2, 2, 2], num_classes)

def test():
    net = ResNet34()
    y = net(torch.randn(4, 1, 64, 64))
    print(y.size())
