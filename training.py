# TRAINING
# =================================================================================================================
# This file contains functions functions that allow the user to train and save a model by storing all the parameters
# into a class called trainer
# =================================================================================================================


# Imports needed for the following code to work
from os import path
import torch
import torch.nn as nn
import torch.optim as optim
import PlotGraph as plt
from tqdm import tqdm
import numpy as np
from torchvision.datasets import ImageFolder
import torch.utils.data as data
from torchvision.transforms import transforms
import matplotlib.pyplot as plot

sys.path.insert(1, 'models')
# ===================================== Import Architectures =====================================

# Imports needed for the following code to work
import AlexNet as AN
import NishNet as NN
import VGG16 as VGG
import ResNet as RN
import LeNet as LN

def outputEmotions(listElement):
    names = ["Afraid", "Angry", "Disgust", "Happy", "Neutral", "Sad", "Surprised"]

    for i in range(len(names)):
        print("\t" + names[i] + ": " + str(listElement[i]))


# trainer -> a class that holds necessary functions/dataset/parameters to train the model and save it to a local device
# Params:
#   -epoch        -> integer input: the amount of times that the network will loop through the entire training dataset
#   -batch_size   -> integer input: the number of images that will be passed into the network at one time
#   -net          -> class input: The chosen network class that will be trained
#   -trainSet     -> list input: training dataset that will be used to train the chosen model
#   -validSet     -> list input: validation dataset that will be used to fine tune the chosen model
#   -testSet      -> list input: testing dataset that will be used for accuracy calculations and plot confusion matrix
#   -device       -> Determines if the calculation will be done on the cpu or gpu in the local device
class trainer:
    def __init__(self, net, device, location, epoch: int, batch_size: int, lr, momentum, wd, factor, imageSize,
                 weights=True, evalMode=False):
        self.net = net
        self.loss_func = None
        self.location = location

        self.batch_size = batch_size
        self.totalEpoch = batch_size
        self.epoch = epoch
        self.image_size = imageSize

        self.trainSet = None
        self.validSet = None
        self.testSet = None
        self.weights = weights
        self.evalMode = evalMode
        self.device = device

        print(self.device)
        self.retrieveData(imageSize, self.location, self.batch_size)
        # ========================================= Detect input errors ================================================
        integers = [epoch, batch_size]
        for i in integers:
            if not isinstance(i, int):
                raise Exception("Invalid input, please check that [EPOCHS] and [BATCH_SIZE] are integers")

        check_list = [self.trainSet, self.validSet, self.testSet]
        check = []
        for i in check_list:
            it = iter(i)
            image, label = next(it)
            image, label = image.to(self.device), label.to(self.device)
            check.append(net(image))
        if check[0].shape == check[1].shape == check[2].shape:
            pass
        else:
            raise Exception("Datasets do not match with each other using current network, please check if:"
                            "[trainSet], [validSet], [testSet] match with each other for ", str(net))
        # ==============================================================================================================

        # Reference for the save and load function
        self.info = [lr, momentum, wd, factor]

        # Initialise the optimiser and the loss function that is being used
        self.optimiser = optim.SGD(self.net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, factor=factor, verbose=True)

        # Initialise the graphing classes that are being used
        self.pltLoss = plt.genLinePlot(title="Loss Analysis", ylabel="Loss", xlabel="Epoch", numOfLines=2,
                                       legendList=["train", "test"])
        self.pltAcc = plt.genLinePlot(title="Accuracy Analysis", ylabel="Accuracy", xlabel="Epoch", numOfLines=2,
                                      legendList=["train", "test"])

    def retrieveData(self, imageSize, location, batch_size):
        # Image augmentation is applied to the processed images
        transformAugmented = transforms.Compose([transforms.Resize(int(imageSize * 1.1)),
                                                 transforms.RandomCrop(imageSize),
                                                 transforms.Grayscale(1),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomAffine(10),
                                                 transforms.ToTensor()])

        train = ImageFolder(location + "\\train", transform=transformAugmented)
        valid = ImageFolder(location + "\\validate", transform=transformAugmented)
        test = ImageFolder(location + "\\test", transform=transformAugmented)
        print("\nIMAGES HAS BEEN RETRIEVED")

        # Find the loss function
        if self.weights and not self.evalMode:
            # Initialize initial weights for network
            print("\nCalculating the weight adjustments for loss...")

            classWeights = torch.zeros((1, 7))
            for _, label in tqdm(train, desc="1/3"):
                classWeights[0][label] += 1

            for _, label in tqdm(valid, desc="2/3"):
                classWeights[0][label] += 1

            for _, label in tqdm(test, desc="3/3"):
                classWeights[0][label] += 1

            classWeights = 1 / classWeights
            classWeights = classWeights.to(self.device)

            self.loss_func = nn.CrossEntropyLoss(weight=classWeights)
        else:
            self.loss_func = nn.CrossEntropyLoss()

        # Load the processed images that are ready for calculation into the program
        self.trainSet = data.DataLoader(train, batch_size=batch_size, shuffle=True)
        self.validSet = data.DataLoader(valid, batch_size=batch_size, shuffle=True)
        self.testSet = data.DataLoader(test, batch_size=batch_size, shuffle=True)
        print("\nIMAGES HAS BEEN LOADED IN THE PROGRAM")

    # trainingEval -> A function that calculates and returns the loss and accuracy at each epoch
    # Params:
    #   -iterations -> integer input: The number of batchs that will be used to compute loss anc accuracy
    #   -train      -> boolean input: determines whether or not it is calculating loss and accuracy for train or
    #                  validation datasets
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

    # evaluateModel -> A function that calculates and returns the loss and accuracy at the end of the training process
    # Params:
    #   -path  -> string input: Location of where the results will be saved to be plotted for later
    #   -name  -> string input: name for the title of the confusion matrix as well as filename for the results
    def evaluateModel(self, path: str, name: str):
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
            for dataTest in tqdm(self.testSet,
                                 desc="Calculating Testing Accuracy",
                                 position=0,
                                 leave=True):

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

            for i in range(7):
                if predictedCount[i] != 0:
                    testPrecision[i] = round(testCorrectDict[i] / predictedCount[i], 3)
                if actualCount[i] != 0:
                    testRecall[i] = round(testCorrectDict[i] / actualCount[i], 3)
                if testPrecision[i] != 0 and testRecall[i] != 0:
                    testF1Score[i] = 2 * ((testRecall[i] * testPrecision[i]) / (testPrecision[i] + testRecall[i]))

            print("""
            |-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-|
                            MODEL SUMMARY AND EVALUATION
            |-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-|
            """)

            print("Test Accuracy : \t" + str(testAcc))
            print("Test Loss :\t" + str(testLoss))
            print("Test Recall :\t")
            outputEmotions(testRecall)
            print("Test Precision :\t")
            outputEmotions(testPrecision)
            print("Test f1 Score :\t")
            outputEmotions(testF1Score)
            print("|-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-|")
            print("CONFUSION MATRIX")

        # Begin the confusion matrix analysis
        empty = torch.zeros(7, 7, dtype=torch.float32)
        confusion_matrix = empty.numpy()
        for i in matrix_plot:
            predicted_emotion, actual_emotion = i
            confusion_matrix[predicted_emotion, actual_emotion] = confusion_matrix[
                                                                      predicted_emotion, actual_emotion] + 1
        print(confusion_matrix)

        plt.plot_confusion_matrix(confusion_matrix, name, path=path, norm=True)

        plt.showPlot(self.pltLoss, path=path, name=name + "-loss")
        plt.showPlot(self.pltAcc, path=path, name=name + "-acc")

        reshapedLoss = np.transpose(self.pltLoss.y)
        reshapedAcc = np.transpose(self.pltAcc.y)

        np.savetxt(path + "/" + name + "-cm.csv", confusion_matrix, delimiter=',')
        np.savetxt(path + "/" + name + "-loss.csv", reshapedLoss, delimiter=',')
        np.savetxt(path + "/" + name + "-acc.csv", reshapedAcc, delimiter=',')

    # startTrain -> A function that trains network by looping through the specifide numbero of epochs and batchs
    # Params:
    #   -Path      -> string input: Location of where the results will be saved to be plotted for later
    #   -FileName  -> string input: file name to load from and save the model for flexible training
    #   -load      -> boolean input: user can decide to load an existing model to continue training or start again
    #   -saveFreq  -> number of epochs to be iterated before triggering a save
    def startTrain(self, Path, fileName="default_model", load=False, saveFreq=20):

        # ========================================= Detect input errors ================================================
        if (not path.exists(Path)) or (not isinstance(Path, str)):
            raise Exception("Yikes! " + "[" + Path + "]" + " Path does not exist :(")

        if not isinstance(fileName, str):
            raise Exception("Invalid type, please make sure [fileName] is a string type")
        # ==============================================================================================================

        if load:
            self.loadCheckpoint(Path, fileName)
            print("\n LOADED IN A NETWORK CHECKPOINT :" + fileName)

        trainLoss = 0
        trainAcc = 0
        validLoss = 0
        validAcc = 0

        # Start the iteration process
        for e in range(self.epoch):
            lastLoss = 0
            for idx, trainingData in tqdm(enumerate(self.trainSet),
                                          desc="EPOCH " + str(e + 1) + "/" + str(self.epoch),
                                          total=len(self.trainSet)):
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

            trainLoss, trainAcc = self.trainingEval(10)
            validLoss, validAcc = self.trainingEval(10, train=False)

            self.scheduler.step(validLoss)
            # self.scheduler.step()

            plt.insertY(self.pltLoss, trainLoss, validLoss)
            plt.insertY(self.pltAcc, trainAcc, validAcc)

            # output message about loss
            print("\nEPOCH #" + str(e + 1) + " Completed.")
            print("Current Loss (Train/Valid): " + str(trainLoss) + "/" + str(validLoss))
            print("Current Accuracy (Train/Valid): " + str(trainAcc) + "/" + str(validAcc) + "\n")

            # Saving a Model at certain intervals of epoch and at the final epoch
            if (e + 1) % saveFreq == 0 or e == (self.epoch - 1):
                self.saveCheckpoint(Path + "/" + fileName + ".pt")
                print("CURRENT MODEL HAS BEEN SAVED AS :" + fileName)

        self.evaluateModel(Path, fileName)

    # saveCheckpoint -> A function that saves the current state of a model and it's results
    # Params:
    #   -fileName      -> string input: Location of where the results will be saved to
    def saveCheckpoint(self, fileName: str):
        checkpoint = {
            "weights_save": self.net.state_dict(),
            "optimizer_save": self.optimiser.state_dict(),
            "scheduler_save": self.scheduler.state_dict(),
            "loss_save": self.loss_func.state_dict(),
            "epoch_save": self.totalEpoch + self.epoch,
            "batchSize_save": self.batch_size,
            "graphData_save_Loss": [self.pltLoss.y, self.pltLoss.x],
            "graphData_save_Acc": [self.pltAcc.y, self.pltAcc.x],
            "info": self.info,
            "imageSize": self.image_size
        }
        torch.save(checkpoint, fileName)

    # loadCheckpoint -> A function that loads back a state of a model that was saved previously from a file location
    # Params:
    #   -checkpoint_path  -> string input: Location of where the saved checkpoint is
    #   -checkpoint_name  -> string input: name of the saved file
    def loadCheckpoint(self, checkpoint_path: str, checkpoint_name: str):
        load_checkpoint = torch.load(checkpoint_path + "/" + checkpoint_name + ".pt")

        self.net.load_state_dict(load_checkpoint["weights_save"])
        self.optimiser.load_state_dict(load_checkpoint["optimizer_save"])
        self.scheduler.load_state_dict(load_checkpoint["scheduler_save"])
        self.totalEpoch = load_checkpoint["epoch_save"]
        self.pltLoss.x = load_checkpoint["graphData_save_Loss"][1]
        self.pltLoss.y = load_checkpoint["graphData_save_Loss"][0]
        self.pltAcc.x = load_checkpoint["graphData_save_Acc"][1]
        self.pltAcc.y = load_checkpoint["graphData_save_Acc"][0]
        self.info = load_checkpoint["info"]

        print("\nMODEL LOADED IN: ")
        print("NAME: " + checkpoint_path + "/" + checkpoint_name + ".pt")
        print("BATCH-SIZE: " + str(load_checkpoint["batchSize_save"]))
        print("EPOCHS TRAINED: " + str(load_checkpoint["epoch_save"]))

        print("\nHYPER-PARAMETER SETTINGS")
        print("Learning Rate: " + str(self.info[0]))
        print("Momentum: " + str(self.info[1]))
        print("Weight Decay: " + str(self.info[1]))
        print("Factor to Decrease by: " + str(self.info[1]) + "\n")
