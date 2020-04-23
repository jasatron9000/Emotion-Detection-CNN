import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PlotGraph as plt
from tdqm import tqdm


# A class that holds the necessary functions and variables to train the data
class trainer:
    def __init__(self, net, trainSet, testSet):
        self.net = net
        self.trainSet = trainSet
        self.testSet = testSet

    def startTrain(self, epoch, batchSize, lr=0.001):
        # Initialise the optimiser and the loss function
        optimiser = optim.Adam(self.net.parameters(), lr=lr)
        loss_func = nn.CrossEntropyLoss()

        # Start the iteration process
        for e in range(epoch):
            for data in tqdm(self.testSet, desc="EPOCH " + str(e + 1) + "/" + str(epoch)):
                batchImage, batchLabel = data

                # Training Algorithm
                optimiser.zero_grad()
                output = self.net(batchImage)
                loss = loss_func(output, batchLabel)
                loss.backward()
                optimiser.step()

            #output message about loss
            print("EPOCH #" + str(e + 1) + " Completed.")
            print("Current Loss: " + str(loss.item()))
