import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PlotGraph as plt
import torch.utils.data as data
from tqdm import tqdm


# A class that holds the necessary functions and variables to train the data
class trainer:
    def __init__(self, net, trainSet, validSet, testSet):
        self.net = net
        self.validSet = validSet
        self.trainSet = trainSet
        self.testSet = testSet

    def startTrain(self, epoch, device, batch, lr=0.001):
        pltLoss = plt.genLinePlot(title="Loss Analysis", ylabel="Loss", xlabel="Epoch", numOfLines=2,
                                  legendList=["train", "test"])
        pltAcc = plt.genLinePlot(title="Accuracy Analysis", ylabel="Accuracy", xlabel="Epoch", numOfLines=2,
                                 legendList=["train", "test"])

        # Initialise the optimiser and the loss function
        optimiser = optim.Adam(self.net.parameters(), lr=lr)
        loss_func = nn.CrossEntropyLoss()

        # Start the iteration process
        for e in range(epoch):

            for trainingData in tqdm(self.trainSet, desc="EPOCH " + str(e + 1) + "/" + str(epoch)):
                batchImage, batchLabel = trainingData
                batchImage = batchImage.to(device)
                batchLabel = batchLabel.to(device)

                # Training Algorithm
                self.net.zero_grad()
                output = self.net(batchImage)
                loss = loss_func(output, batchLabel)
                loss.backward()
                optimiser.step()

            # Validation Accuracy/Loss Calculation
            with torch.no_grad():
                validCorrect = 0
                validSum = 0
                trainCorrect = 0
                trainSum = 0

                # For the Training
                for dataTrain in tqdm(self.trainSet, desc="Calculating Training Loss/Accuracy"):
                    trainImage, trainLabel = dataTrain

                    trainImage = trainImage.to(device)
                    trainLabel = trainLabel.to(device)
                    trainOutputAcc = self.net(trainImage)

                    for index, i in enumerate(trainOutputAcc):
                        if torch.argmax(i) == trainLabel[index]:
                            trainCorrect += 1
                    break

                trainAcc = validCorrect / batch
                trainLoss = validSum / batch

                # For the Validation
                for dataValid in tqdm(self.validSet, desc="Calculating Validation Loss/Accuracy"):
                    validImage, validLabel = dataValid

                    validImage = validImage.to(device)
                    validLabel = validLabel.to(device)

                    validOutputAcc = self.net(validImage)

                    for index, i in enumerate(validOutputAcc):
                        validSum += loss_func(validOutputAcc, validLabel).item()

                        if torch.argmax(i) == validLabel[index]:
                            validCorrect += 1
                    break

                validAcc = validCorrect / batch
                validLoss = validSum / batch

            plt.insertY(pltLoss, trainLoss, validLoss)
            plt.insertY(pltAcc, trainAcc, validAcc)

            # output message about loss
            print("\nEPOCH #" + str(e + 1) + " Completed.")
            print("Current Loss (Train/Valid): " + str(trainLoss) + "/" + str(validLoss))
            print("Current Accuracy (Train/Valid): " + str(trainAcc) + "/" + str(validAcc) + "\n")

        plt.showPlot(pltLoss, pltAcc)