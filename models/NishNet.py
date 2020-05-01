import torch
import torch.nn as nn
import torch.nn.functional as F

output_features = 64
conv_dropout_p = 0.5
fc_dropout_p = 0.4
num_labels = 7

class NishNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 1 input image channel, 64  feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, output_features, 3, padding=(1, 1))
        self.conv2 = nn.Conv2d(output_features, output_features, 3, padding=(1, 1))
        self.BN1 = nn.BatchNorm2d(output_features)

        self.conv3 = nn.Conv2d(output_features, output_features * 2, 3, padding=(1, 1))
        self.BN2 = nn.BatchNorm2d(output_features * 2)
        self.conv4 = nn.Conv2d(output_features * 2, output_features * 2, 3, padding=(1, 1))

        self.conv5 = nn.Conv2d(output_features * 2, output_features * 2 * 2, 3, padding=(1, 1))
        self.BN3 = nn.BatchNorm2d(output_features * 2 * 2)
        self.conv6 = nn.Conv2d(output_features * 2 * 2, output_features * 2 * 2, 3, padding=(1, 1))

        self.conv7 = nn.Conv2d(output_features * 2 * 2, output_features * 2 * 2 * 2, 3, padding=(1, 1))
        self.BN4 = nn.BatchNorm2d(output_features * 2 * 2 * 2)
        self.conv8 = nn.Conv2d(output_features * 2 * 2 * 2, output_features * 2 * 2 * 2, 3, padding=(1, 1))

        self.conv_layers = {
            1: self.conv1,
            2: self.conv2,
            3: self.conv3,
            4: self.conv4,
            5: self.conv5,
            6: self.conv6,
            7: self.conv7,
            8: self.conv8
        }
        self.norm_layers = {
            1: self.BN1,
            2: self.BN2,
            3: self.BN3,
            4: self.BN4
        }

        self.fc1 = nn.Linear(4608, output_features * 2 * 2 * 2)
        self.fc2 = nn.Linear(output_features * 2 * 2 * 2, output_features * 2 * 2)
        self.fc3 = nn.Linear(output_features * 2 * 2, output_features * 2)
        self.fc4 = nn.Linear(output_features * 2, num_labels)

    def convs(self, x, conv_layer: int, norm_layer: int):
        x = F.relu(self.conv_layers[conv_layer](x))
        x = self.norm_layers[norm_layer](x)
        x = F.relu(self.conv_layers[conv_layer + 1](x))
        x = self.norm_layers[norm_layer](x)
        x = F.dropout(F.max_pool2d(x, (2, 2), 2), p=conv_dropout_p, training=True)

        return x

    def forward(self, x):
        # pass through the conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.BN1(x)
        x = F.dropout(F.max_pool2d(x, (2, 2), 2), p=conv_dropout_p, training=True)

        x = self.convs(x, 3, 2)
        x = self.convs(x, 5, 3)
        x = self.convs(x, 7, 4)

        # flatten
        x = x.view(-1, 4608)

        # pass through the FC layers
        x = F.dropout(F.relu(self.fc1(x)), p=fc_dropout_p, training=True)
        x = F.dropout(F.relu(self.fc2(x)), p=fc_dropout_p, training=True)
        x = F.dropout(F.relu(self.fc3(x)), p=fc_dropout_p, training=True)
        x = self.fc4(x)

        return F.softmax(x, dim=1)