import os
import cv2

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision
from torchvision import transforms, datasets

#------------------------------------------------ LeNet Model class -------------------------------------------------#
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 1 input image channel, 6  feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        x = torch.randn(32, 32).view(-1, 1, 32, 32)
        self.to_linear = None

        # Calling function that only runs through the convolution layers
        self.convs(x)

        self.fc1 = nn.Linear(self.to_linear, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # a function that ONLY runs through the convolution layers to get dimensions
    # as input for the FC layers
    def convs(self, x):
        x = F.avg_pool2d(torch.tanh(self.conv1(x)), (2, 2), 2)
        x = F.avg_pool2d(torch.tanh(self.conv2(x)), (2, 2), 2)

        #         print(x[0].shape)

        #         after the first 2 conv layers we find the shape of it
        #         x[0].shape[0]= number of feature maps
        #         x[0].shape[1]= dimension
        #         x[0].shape[2]= dimension
        #        in this case self._to_linear = 16 * 5 * 5
        if self.to_linear is None:
            self.to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)  # run through conv layers
        x = x.view(-1, self.to_linear)  # flatten

        # pass throught the FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #         print(x[0].shape)
        x = self.fc3(x)

        return F.softmax(x, dim=1)


net = LeNet()

#------------------------------------------------ Optimisers/Loss function used -------------------------------------------------#
optimizer = optim.Adam(net.parameters(), lr = 0.001)
loss_function = nn.CrossEntropyLoss()


#------------------------------------------------ seperating the images from answers -------------------------------------------------#
# self.training_data.append([np.array(img), np.eye(7)[self.labels[emotion]]])
X = torch.Tensor([i[0] for i in training_data]).view(-1, 32, 32)
X = X/255.0 # make the colour values between 0 and 1 instead of 0 to 255
y = torch.Tensor([i[1] for i in training_data])


#------------------------------------------------ Selecting testing/validation ratio of dataset -------------------------------------------------#
percent_valid = 0.2  # using 30% dataset
valid_size = int(len(X) * percent_valid)
# [                <--valid_size         ]
train_X = X[:-valid_size]
train_y = y[:-valid_size]
# [                valid_size-->         ]
test_X = X[-valid_size:]
test_y = y[-valid_size:]


#------------------------------------------------ Selection of batch size and epochs -------------------------------------------------#
batch_size = 5
epochs = 100

#------------------------------------------------ Training LeNet -------------------------------------------------#
for epoch in tqdm(range(epochs)):
    # range(start, end, step_size)
    for i in range(0, len(train_X), batch_size):

        batch_X = train_X[i: i + batch_size].view(-1, 1, 32, 32)
        batch_y = []
        for img in range(batch_X.size()[0]):
            batch_y.append(torch.argmax(train_y[img]).item())

        newBatch_y = torch.FloatTensor(batch_y).long()

        # can use optimizer.zero_grad() if there are multiple optimizers
        # being used
        optimizer.zero_grad()
        output = net(batch_X)
        loss = loss_function(output, newBatch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch}. Loss: {loss}")

#------------------------------------------------ MIGHT BE REPLACED WITH SEPERATE FUNCTION -------------------------------------------------#
#------------------------------------------------------- Accuracy Calculation --------------------------------------------------------#
correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):

        # torch.argmax Returns the indices of the maximum value
        # of all elements in the input tensor.
        # checks location of hot 1
        actual_emotion = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1, 1, 32, 32))[0]
        # checks location of softmax
        predicted_emotion = torch.argmax(net_out)

        if predicted_emotion == actual_emotion:
            correct += 1
        total += 1
print("Accuracy", round(correct / total, 3))