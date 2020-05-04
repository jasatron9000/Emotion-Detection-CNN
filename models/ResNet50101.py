import torch
import torch.nn as nn
import torch.nn.functional as F


class bottleNeck_block(nn.Module):
    # 3 conv layers: kernal 1 -> 3 -> 1

    # For bottle neck block design, there is always a x4 expansion to the number of output layers
    # for the last conv layer in the block
    blk_expand = 4

    def __init__(self, in_channels, map_back_channels, change_residual_shape=None, stride=1):
        super(bottleNeck_block, self).__init__()

        # The first conv layer will take in the in_channel size which will always be
        # x4 the map_back_layer accept for in the beginning
        #
        # Everytime the in_channels will be mapped back to it's original size before the expansion
        # as the block must be repeated when making the ResNet layers
        self.conv1 = nn.Conv2d(in_channels, map_back_channels, kernel_size=1, stride=stride, padding=0)
        self.BN1 = nn.BatchNorm2d(map_back_channels)

        self.conv2 = nn.Conv2d(map_back_channels, map_back_channels, kernel_size=3, stride=1, padding=1)
        self.BN2 = nn.BatchNorm2d(map_back_channels)

        self.conv3 = nn.Conv2d(map_back_channels, map_back_channels * bottleNeck_block.blk_expand, kernel_size=1,
                               stride=1, padding=0)
        self.BN3 = nn.BatchNorm2d(map_back_channels * bottleNeck_block.blk_expand)

        # If a change_residual_shape was given as a input then it means the map_back_channels
        # has been changed(going into the next ResNet layer)
        #
        # Therefore the Sequential for changing the residual to match the current output(x)
        # must be updated
        self.change_residual_shape = change_residual_shape

    def forward(self, x):
        # The residual will be assigned at the start of each block to then be added at the end
        Residual = x

        # Go through the bottleNeck_block conv layers
        x = self.conv1(x)
        x = F.relu(self.BN1(x))

        x = self.conv2(x)
        x = F.relu(self.BN2(x))

        x = self.conv3(x)
        x = F.relu(self.BN3(x))

        # changes the residual dimensions to match the current output so that it can be added
        # to the current output
        if self.change_residual_shape is not None:
            Residual = self.change_residual_shape(Residual)

        # Adds on the residual
        x = x + Residual

        x = F.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, bottleNeck_block, layers, num_classes):
        super(ResNet, self).__init__()

        # The initial convolution layer before the ResNet layers thats made up of the bottleNeck_blocks
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.BN1 = nn.BatchNorm2d(64)

        # nn.functions were not used as resnet uses specific parameters for pooling, therefore
        # the 2 different types(max and adaptive_avg is specifide here)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Defining the Resnet layers that will call to a function that will create a Sequential(pytorch function)
        # that will string together the right number of bottleNeck_blocks depending on which version of resnet
        # that is being used is.
        #
        # The map_back_channels are specific to the resnet architecture and is what the in_channel
        # will be mapped back to within every bottleNeck_block depending on the layer
        # The difference is always 2 time the previous layer.
        # 64 -> 128 -> 256 -> 521
        #
        # The stride is hard coded to 2 in the make_layer function as it is only 1 initially for the first layer
        # The stride will be used in the very first bottleNeck_block in a layer to reduce the dimension,
        # this will happen throughout the network.
        # Inside bottleNeck_block: stride: 2 -> 1 -> 1
        self.resLayer1 = self.make_layer(bottleNeck_block, layers[0], map_back_channels=64, stride=1)
        self.resLayer2 = self.make_layer(bottleNeck_block, layers[1], map_back_channels=128)
        self.resLayer3 = self.make_layer(bottleNeck_block, layers[2], map_back_channels=256)
        self.resLayer4 = self.make_layer(bottleNeck_block, layers[3], map_back_channels=512)

        # The fully connected layer will be the last map_back_channels value * 4 from the last conv layer
        self.fc = nn.Linear(512 * bottleNeck_block.blk_expand, num_classes)

    def make_layer(self, bottleNeck_block, num_blocks, map_back_channels, stride=2):

        # Initialise it as None to be assigned later on
        change_residual_shape = None

        # Initialise an empty list that will hold the required number of bottleNeck_block objects
        # for the particular layer
        layer_blocks = []

        # Defining the conv and batch layer that will be used to change the dimension of the residual
        #
        # This will be in a pytorch Sequential and assigned to change_residual_shape when a change in
        # map_back_channels is detected
        self.resConv = nn.Conv2d(self.in_channels, map_back_channels * bottleNeck_block.blk_expand, kernel_size=1,
                                 stride=stride)
        self.resBatch = nn.BatchNorm2d(map_back_channels * bottleNeck_block.blk_expand)

        # Because the in_channel should always be 4 times the size of the map_back_channels in the
        # bottleNeck_block, if it is no longer the case then it means map_back_channels value has increased
        # and it is the start of a new ResNet layer
        if stride != 1 or self.in_channels != map_back_channels * bottleNeck_block.blk_expand:
            change_residual_shape = nn.Sequential(self.resConv,
                                                  self.resBatch)

        # A initial block will be appended to the the list first as that block will have the stride value that is
        # giving when making the layers, which is in charge of decreasing the the dimension on the feature maps(by half)
        layer_blocks.append(bottleNeck_block(self.in_channels, map_back_channels, change_residual_shape, stride))

        # The in_channel value must also update at the start of every layer as it should remain to be x4 the map_back_channels
        self.in_channels = map_back_channels * bottleNeck_block.blk_expand

        # A for loop is used (in range num_blocks - 1 because already appended previously) to
        # add on the rest of the bottleNeck_blocks for the layer(stride is already hard coded to 1 in the bottleNeck_block class)
        for i in range(num_blocks - 1):
            layer_blocks.append(bottleNeck_block(self.in_channels, map_back_channels))

        # Finally returns a pytorch Sequential that strings together all the bottleNeck blocks together
        return nn.Sequential(*layer_blocks)

    def forward(self, x):
        # Going through initial layer BEFORE using the ResNet layers that are made up of bottleneck blocks
        x = self.conv1(x)
        x = F.relu(self.BN1(x))
        x = self.maxpool(x)

        # Going through the ResNet layers
        x = self.resLayer1(x)
        x = self.resLayer2(x)
        x = self.resLayer3(x)
        x = self.resLayer4(x)

        # finally to a fully connected layer with 512 * blk_expansion connections
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

#------------------------------------- ResNet selection -------------------------------------------#

# img_channel: 1 (grayscale) or 3 (RGB)
def ResNet50():
    num_classes = 7
    return ResNet(bottleNeck_block, [3, 4, 6, 3], num_classes)

def ResNet101():
    num_classes = 7
    return ResNet(bottleNeck_block, [3, 4, 23, 3], num_classes)

def ResNet152():
    num_classes = 7
    return ResNet(bottleNeck_block, [3, 8, 36, 3], num_classes)
