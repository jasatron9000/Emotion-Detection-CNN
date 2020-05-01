import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PlotGraph as plt
import torch.utils.data as data
from tqdm import tqdm


# A class that holds the necessary functions and variables to train the data
class trainer:
    def __init__(self, epoch, batch_size, net, trainSet, validSet, testSet, device, lr=0.005):
        self.net = net
        self.validSet = validSet
        self.trainSet = trainSet
        self.testSet = testSet
        self.device = device

        self.batch_size = batch_size
        self.epoch = epoch

        # Initialise the optimiser and the loss function that is being used
        self.optimiser = optim.SGD(self.net.parameters(), lr=lr)
        self.loss_func = nn.CrossEntropyLoss()

        # Initialise the graphing classes that are being used
        self.pltLoss = plt.genLinePlot(title="Loss Analysis", ylabel="Loss", xlabel="Epoch", numOfLines=2,
                                       legendList=["train", "test"])
        self.pltAcc = plt.genLinePlot(title="Accuracy Analysis", ylabel="Accuracy", xlabel="Epoch", numOfLines=2,
                                      legendList=["train", "test"])

    def trainingEval(self, iterations, train=True):
        with torch.no_grad():
            sumLoss = 0
            correct = 0
            total = 0

            if train:
                it = iter(self.trainSet)
            else:
                it = iter(self.validSet)

            for iterate in range(iterations):
                image, label = next(it)
                image = image.to(self.device)
                label = label.to(self.device)

                # Calculate the loss
                outputEval = self.net(image)
                sumLoss += self.loss_func(outputEval, label).item()

                for idx, out in enumerate(outputEval):
                    if torch.argmax(out) == label[idx]:
                        correct += 1
                    total += 1

            lossOut = sumLoss / iterations
            accOut = correct / total

            return lossOut, accOut

    def evaluateModel(self):
        testCorrectDict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        predictedCount = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        actualCount = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        testRecall = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        testPrecision = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        testF1Score = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        testCorrect = 0
        testSum = 0
        matrix_plot = []
        count = 0
        count2 = 0
        # Test set loop process / Retrieves the average loss, accuracy, recall, precision and f1 score
        with torch.no_grad():
            for dataTest in tqdm(self.testSet, desc="Calculating Testing Accuracy", position=0, leave=True):
                testImage, testLabel = dataTest
                testImage = testImage.to(self.device)
                testLabel = testLabel.to(self.device)
                predicted = self.net(testImage)

                testSum += self.loss_func(predicted, testLabel).item()
                count += 1

                for index, i in enumerate(predicted):
                    matrix_plot.append([torch.argmax(i), testLabel[index]])

                    predictedCount[torch.argmax(i).item()] += 1
                    actualCount[testLabel[index].item()] += 1

                    if torch.argmax(i) == testLabel[index]:
                        testCorrect += 1
                        testCorrectDict[testLabel[index].item()] += 1
                    count2 += 1

            testAcc = testCorrect / count2
            testLoss = testSum / count

            print(testCorrectDict)
            print(predictedCount)
            print(actualCount)

            for i in range(7):
                if predictedCount[i] != 0:
                    testRecall[i] = round(testCorrectDict[i] / predictedCount[i], 3)
                if testPrecision[i] != 0:
                    testPrecision[i] = round(testCorrectDict[i] / actualCount[i], 3)
                if predictedCount[i] != 0 and testPrecision[i] != 0:
                    testF1Score[i] = 2 * ((testRecall[i]*testPrecision[i])/(testPrecision[i]+testRecall[i]))

            print("""
            |-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-|
                            MODEL SUMMARY AND EVALUATION
            |-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-|
            """)

            print("Test Accuracy : \t" + str(testAcc))
            print("Test Loss :\t" + str(testLoss))
            print("Test Recall :\t" + str(testRecall))
            print("Test Precision :\t" + str(testPrecision))
            print("Test f1 Score :\t" + str(testF1Score))
            print("|-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-|")
            print("CONFUSION MATRIX")

        # Begin the confusion matrix analysis
        empty = torch.zeros(7, 7, dtype=torch.int32)
        confusion_matrix = empty.numpy()
        for i in matrix_plot:
            predicted_emotion, actual_emotion = i
            confusion_matrix[predicted_emotion, actual_emotion] = confusion_matrix[
                                                                      predicted_emotion, actual_emotion] + 1
        print(confusion_matrix)
        plt.plot_confusion_matrix(confusion_matrix, "TestClass")

        plt.showPlot(self.pltLoss)
        plt.showPlot(self.pltAcc)

    def startTrain(self, load=False, fileName="default_model", saveFreq=20):
        if load:
            self.loadCheckpoint(fileName)
            print("\n LOADED IN A NETWORK CHECKPOINT :" + fileName)

        trainLoss = 0
        trainAcc = 0
        validLoss = 0
        validAcc = 0

        # Start the iteration process
        for e in range(self.epoch):
            lastLoss = 0
            for idx, trainingData in tqdm(enumerate(self.trainSet), desc="EPOCH " + str(e + 1) + "/" + str(self.epoch),
                                          position=0, leave=True, total=len(self.trainSet)):
                batchImage, batchLabel = trainingData
                batchImage = batchImage.to(self.device)
                batchLabel = batchLabel.to(self.device)

                # Training Algorithm
                self.net.zero_grad()
                output = self.net(batchImage)
                loss = self.loss_func(output, batchLabel)
                lastLoss = loss.item()
                loss.backward()
                self.optimiser.step()

                # torchtest.assert_vars_change(self.net,
                #                              self.loss_func,
                #                              self.optimiser,
                #                              batch=[batchImage, batchLabel],
                #                              device=self.device)

            trainLoss, trainAcc = self.trainingEval(5)
            validLoss, validAcc = self.trainingEval(5, train=False)

            plt.insertY(self.pltLoss, trainLoss, validLoss)
            plt.insertY(self.pltAcc, trainAcc, validAcc)

            # output message about loss
            print("\nEPOCH #" + str(e + 1) + " Completed.")
            print("Current Loss (Train/Valid): " + str(trainLoss) + "/" + str(validLoss))
            print("Current Accuracy (Train/Valid): " + str(trainAcc) + "/" + str(validAcc) + "\n")

            # Saving a Model at certain intervals of epoch and at the final epoch
            if (e + 1) % saveFreq == 0 or e == (self.epoch - 1):
                self.saveCheckpoint(e + 1, self.batch_size, fileName)
                print("CURRENT MODEL HAS BEEN SAVED AS :" + fileName)

        self.evaluateModel()

    # Saves the progress to a fileName that was specified
    def saveCheckpoint(self, fileName: str, save_dir = None):
        checkpoint = {
            "model_save": self.net.state_dict(),
            "optimizer_save": self.optimiser.state_dict(),
            "epoch_save": self.epoch,
            "batchSize_save": self.batch_size,
            "graphData_save_Loss": [self.pltLoss.y, self.pltLoss.x],
            "graphData_save_Acc": [self.pltAcc.y, self.pltAcc.x]
        }
        torch.save(checkpoint, fileName)
        if save_dir is not None:
            shutil.copy(fileName, dst=save_dir)

    # Loads in the progress that was made
    def loadCheckpoint(self, checkpoint_path: str):
        load_checkpoint = torch.load(checkpoint_path)

        self.net.load_state_dict(load_checkpoint["model_save"])
        self.optimiser.load_state_dict(load_checkpoint["optimizer_save"])
        self.pltLoss.x = load_checkpoint["model_save"][1]
        self.pltLoss.y = load_checkpoint["model_save"][0]
        self.pltAcc.x = load_checkpoint["model_save"][1]
        self.pltAcc.y = load_checkpoint["model_save"][0]
        self.batch_size = load_checkpoint["batchSize_save"]
        self.epoch = load_checkpoint["epoch_save"]
