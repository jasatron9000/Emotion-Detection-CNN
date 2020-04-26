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

    def startTrain(self, epoch, batchSize, lr=0.001, accCalc=10):
        pltLoss = plt.genLinePlot(title="Loss Analysis", ylabel="Loss", xlabel="Epoch", numOfLines=2,
                                  legendList=["train", "test"])
        pltAcc = plt.genLinePlot(title="Accuracy Analysis", ylabel="Accuracy", xlabel="Epoch", numOfLines=2,
                                 legendList=["train", "test"])

        # Initialise the optimiser and the loss function
        optimiser = optim.Adam(self.net.parameters(), lr=lr)
        loss_func = nn.CrossEntropyLoss()

        # Start the iteration process
        for e in range(epoch):

            for index, data in tqdm(enumerate(self.trainSet), desc="EPOCH " + str(e + 1) + "/" + str(epoch)):
                batchImage, batchLabel = data

                # Training Algorithm
                optimiser.zero_grad()
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
                for data in tqdm(range(len(self.trainSet)), desc="Calculating Training Loss/Accuracy"):
                    trainImage, trainLabel = data
                    trainOutputAcc = self.net(validImage)

                    for index, i in enumerate(trainOutputAcc):
                        trainSum += loss_func(trainOutputAcc, trainLabel).item()

                        if torch.argmax(i) == trainLabel[index]:
                            trainCorrect += 1

                trainAcc = validCorrect / len(self.trainSet)
                trainLoss = validSum / len(self.trainSet)

                # For the Validation
                for data in tqdm(range(len(self.validSet)), desc="Calculating Validation Loss/Accuracy"):
                    validImage, validLabel = data
                    validOutputAcc = self.net(validImage)

                    for index, i in enumerate(validOutputAcc):
                        validSum += loss_func(validOutputAcc, validLabel).item()

                        if torch.argmax(i) == validLabel[index]:
                            validCorrect += 1

                validAcc = validCorrect / len(self.validSet)
                validLoss = validSum / len(self.validSet)


            plt.insertY(pltLoss, trainLoss, validLoss)
            plt.insertY(pltAcc, trainAcc, validAcc)

            # output message about loss
            print("EPOCH #" + str(e + 1) + " Completed.")
            print("Current Class")

        TestCorrect = 0
        TestSum = 0
        matrix_plot = []
        with torch.no_grad():
            for data in tqdm(range(len(self.testSet)), desc="Calculating Testing Accuracy"):
                testImage, testLabel = data
                predicted = self.net(testImage)

                for index, i in enumerate(predicted):
                    testSum += loss_func(testOutputAcc, testLabel).item()
                    matrix_plot.append([torch.argmax(i), testLabel[index]])

                    if torch.argmax(i) == testLabel[index]:
                        testCorrect += 1

            testAcc = testCorrect / len(self.testSet)
            testLoss = testSum / len(self.testSet)

        empty = torch.zeros(7, 7, dtype=torch.int32)
        confusion_matrix = confusion_matrix.numpy()
        for i in matrix_plot:
            predicted_emotion, actual_emotion = i
            empty[predicted_emotion, actual_emotion] = confusion_matrix[predicted_emotion, actual_emotion] + 1


class loadTensors(data.Dataset):
    def __init__(self, images, labels):
        # Convert the training data of
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return [self.images[index], self.labels[index]]
