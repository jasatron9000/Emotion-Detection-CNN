import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PlotGraph as plt
import torch.utils.data as data
from tqdm import tqdm


# A class that holds the necessary functions and variables to train the data
class trainer:
    def __init__(self, net, trainSet, validSet, testSet, lr=0.001):
        self.net = net
        self.validSet = validSet
        self.trainSet = trainSet
        self.testSet = testSet

        # Initialise the optimiser and the loss function that is being used
        self.optimiser = optim.Adam(self.net.parameters(), lr=lr)
        self.loss_func = nn.CrossEntropyLoss()

    def startTrain(self, epoch, device, batch, load=False, fileName="-", saveFreq=20):
        pltLoss = plt.genLinePlot(title="Loss Analysis", ylabel="Loss", xlabel="Epoch", numOfLines=2,
                                  legendList=["train", "test"])
        pltAcc = plt.genLinePlot(title="Accuracy Analysis", ylabel="Accuracy", xlabel="Epoch", numOfLines=2,
                                 legendList=["train", "test"])

        if load:
            self.loadCheckpoint(fileName)
            print("\n LOADED IN A NETWORK CHECKPOINT :" + fileName)

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
                loss = self.loss_func(output, batchLabel)
                lastLoss = loss.item()
                loss.backward()
                self.optimiser.step()

            # Validation Accuracy/Loss Calculation
            with torch.no_grad():
                validCorrect = 0
                validSum = 0
                trainCorrect = 0
                trainSum = 0

                # For the Training
                for dataTrain in tqdm(self.trainSet,
                                      desc="Calculating Training Loss/Accuracy"):
                    trainImage, trainLabel = dataTrain

                    trainImage = trainImage.to(device)
                    trainLabel = trainLabel.to(device)
                    trainOutputAcc = self.net(trainImage)

                    for index, i in enumerate(trainOutputAcc):
                        if torch.argmax(i) == trainLabel[index]:
                            trainCorrect += 1
                    break

                trainAcc = trainCorrect / batch
                trainLoss = lastLoss

                # For the Validation
                for dataValid in tqdm(self.validSet, desc="Calculating Validation Loss/Accuracy"):
                    validImage, validLabel = dataValid

                    validImage = validImage.to(device)
                    validLabel = validLabel.to(device)

                    validOutputAcc = self.net(validImage)

                    for index, i in enumerate(validOutputAcc):
                        validSum += self.loss_func(validOutputAcc, validLabel).item()

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

            # Saving a Model at certain intervals of epoch and at the final epoch
            if (e + 1) % saveFreq == 0 or e == (epoch - 1):
                self.saveCheckpoint(e + 1, batch, fileName)
                print("CURRENT MODEL HAS BEEN SAVED AS :" )

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
                    testSum += self.loss_func(predicted, testLabel).item()
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

    # Saves the progess to a fileName that was specified
    def saveCheckpoint(self, epochCurrent: int, batchSize: int, fileName: str):
        checkpoint = {
            "model_save": self.net.state_dict(),
            "optimizer_save": self.optimiser.state_dict(),
            "epoch_save": epochCurrent,
            "batchSize_save": batchSize
        }

        torch.save(checkpoint, fileName)

    # Loads in the progress that was made
    def loadCheckpoint(self, checkpoint_path: str):
        load_checkpoint = torch.load(checkpoint_path)

        self.net.load_state_dict(load_checkpoint["model_save"])
        self.optimiser.load_state_dict(load_checkpoint["optimizer_save"])

