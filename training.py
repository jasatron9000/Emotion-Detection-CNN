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

        self.loss_func =  nn.CrossEntropyLoss()

    def startTrain(self, epoch, device, batch, lr=0.001):
        pltLoss = plt.genLinePlot(title="Loss Analysis", ylabel="Loss", xlabel="Epoch", numOfLines=2,
                                  legendList=["train", "test"])
        pltAcc = plt.genLinePlot(title="Accuracy Analysis", ylabel="Accuracy", xlabel="Epoch", numOfLines=2,
                                 legendList=["train", "test"])

        # Initialise the optimiser and the loss function
        optimiser = optim.Adam(self.net.parameters(), lr=lr)



        # Start the iteration process
        for e in range(epoch):
            lastLoss = 0

            for trainingData in tqdm(self.trainSet, desc="EPOCH " + str(e + 1) + "/" + str(epoch)):
                batchImage, batchLabel = trainingData
                batchImage = batchImage.to(device)
                batchLabel = batchLabel.to(device)

                # Training Algorithm
                self.net.zero_grad()
                output = self.net(batchImage)
                loss = loss_func(output, batchLabel)
                lastLoss = loss.item()
                loss.backward()
                optimiser.step()

            plt.insertY(pltLoss, trainLoss, validLoss)
            plt.insertY(pltAcc, trainAcc, validAcc)

            # output message about loss
            print("\nEPOCH #" + str(e + 1) + " Completed.")
            print("Current Loss (Train/Valid): " + str(trainLoss) + "/" + str(validLoss))
            print("Current Accuracy (Train/Valid): " + str(trainAcc) + "/" + str(validAcc) + "\n")

        testCorrect = 0
        testSum = 0
        matrix_plot = []
        with torch.no_grad():
            for dataTest in tqdm(self.testSet, desc="Calculating Testing Accuracy"):
                testImage, testLabel = dataTest
                testImage = testImage.to(device)
                testLabel = testLabel.to(device)
                predicted = self.net(testImage)

                for index, i in enumerate(predicted):
                    testSum += loss_func(predicted, testLabel).item()
                    matrix_plot.append([torch.argmax(i), testLabel[index]])

                    if torch.argmax(i) == testLabel[index]:
                        testCorrect += 1

            testAcc = testCorrect / len(self.testSet)
            testLoss = testSum / len(self.testSet)

        empty = torch.zeros(7, 7, dtype=torch.int32)
        confusion_matrix = empty.numpy()
        for i in matrix_plot:
            predicted_emotion, actual_emotion = i
            confusion_matrix[predicted_emotion, actual_emotion] = confusion_matrix[
                                                                      predicted_emotion, actual_emotion] + 1

        plt.plot_confusion_matrix(confusion_matrix, "TestClass")

        plt.showPlot(pltLoss, pltAcc)
        print(confusion_matrix)

        def trainingEval(iterations, train=False):
            with self.net.zero_grad():
                sumLoss = 0
                correct = 0
                total = 0

                if train:
                    it = iter(self.trainSet)
                else:
                    it = iter(self.validSet)

                for iterate in range(iterations):
                    image, label = next(it)

                    # Calculate the loss
                    outputEval = self.net(image)
                    sumLoss += self.loss_func(output, label).item()

                    for idx, out in enumerate(outputEval):
                        if torch.argmax(out) == label[idx]:
                            correct += 1
                        total += 1

                lossOut = sumLoss / iterations
                accOut = correct / total

                return lossOut, accOut




