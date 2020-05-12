# imports
import sys
import os

sys.path.insert(1, 'models')
# ===================================== Import Architectures =====================================

# Imports needed for the following code to work

import NishNet as NN
import VGG16 as VGG
import ResNet as RN
import LeNet as LN

class userInput:
    def __init__(self):
        # Store values that are used in the main.py
        # File names
        self.SAVE = False
        self.DATA_REBUILD = False
        self.DATA_LOCATION = "data"
        self.SAVE_LOCATION = "edited"
        self.MODEL_SAVE = "saved"
        self.MODEL_NAME = ""

        # Hyper-Parameter Settings
        self.MOMENTUM = 0.9
        self.WEIGHT_DECAY = 0.0005
        self.LR = 0.01
        self.BATCH_SIZE = 64
        self.EPOCHS = 100
        self.IMAGE_SIZE = -1
        self.FACTOR = 0.1
        self.TRAIN_PERCENT = 0.7

        # Model
        self.SELECTED_MODEL = None
        self.FUNCTION = -1;

    # Function that deals with error handling with regards of choosing a range between x1 -> x2
    # Returns true or false
    @staticmethod
    def askRange(start, end, question: str, reply=None):
        questionAsked = False
        if reply is None:
            reply = question

        while True:
            try:
                if not questionAsked:
                    questionAsked = True
                    rng = int(input(question))
                else:
                    rng = int(input(reply))
            except ValueError or TypeError:
                print("Invalid string input, please enter an integer value \n")
                continue

            if not start <= rng <= end:
                print("Invalid input, please enter value between 1 and {}\n".format(end))
                continue
            else:
                return rng

    # Function that asks user for an integer where the negative is true when the program wants something greater than
    # 0
    @staticmethod
    def askDigit(question: str, reply=None, negative=True):
        questionAsked = False
        if reply is None:
            reply = question

        while True:
            try:
                if not questionAsked:
                    questionAsked = True
                    dgt = int(input(question))
                else:
                    dgt = int(input(reply))
            except ValueError or TypeError:
                print("Invalid string input, please enter an integer value \n")
                continue

            if dgt <= 0 and negative:
                print("Invalid input, please enter value greater than 0\n")
                continue
            else:
                return dgt

    @staticmethod
    # Checks if the user gave a float as an answer
    def askFloat(question: str, reply=None, negative=True):
        questionAsked = False
        if reply is None:
            reply = question

        while True:
            try:
                if not questionAsked:
                    questionAsked = True
                    dgt = float(input(question))
                else:
                    dgt = float(input(reply))
            except ValueError or TypeError:
                print("Invalid string input, please enter an integer value \n")
                continue

            if dgt <= 0 and negative:
                print("Invalid input, please enter value greater than 0\n")
                continue
            else:
                return dgt

    # Function that detects asks the user yes or no and outputs true or false
    @staticmethod
    def askYesNo(question: str):
        answer = input(question)

        while answer != "y" and answer != "n":
            print("Invalid input, please enter either 'y' or 'n' \n")
            answer = input("(y/n): ")

        return answer == "y"

    # Function that asks user the necessary information to operate the program
    def initValues(self):
        # A
        models = {
            1: LN.LeNet(),
            2: NN.NishNet(),
            3: VGG.CustomVGG13(),
            4: RN
        }

        imageSize = {
            1: 32,
            2: 64,
            3: 64,
            4: 64
        }

        ResNet = {
            1: RN.ResNet50(),
            2: RN.ResNet101(),
            3: RN.ResNet152(),
            4: RN.ResNet34(),
            5: RN.ResNet18(),
            6: RN.ResNet56()
        }

        # ========================== USER INPUT ========================== #
        print(
            """
            +=======================================================+
            |                                                       |
            |                EMOTION RECOGNITION CNN                |
            |                                                       |
            |       Created by: Jason Buquiran and Shiyao Wang      |
            |                                                       |
            +=======================================================+

            """
        )
        #Asks the user to input which settings they want to do
        choice = self.askRange(1, 3, "Select which tasks you would like to choose: \n"
                                     "1. Train new network \n"
                                     "2. Train pre-existing network \n"
                                     "3. Evaluate pre-existing network \n"
                                     "\n"
                                     "CHOICE: ")
        self.FUNCTION = choice
        # Choose a network to start training on
        modelOption = self.askRange(1, len(models),
                                    "Please select Architecture to use, enter a number to select the model: \n"
                                    "1. LeNet-5 \n"
                                    "2. VGG11 \n"
                                    "3. VGG13 \n"
                                    "4. ResNet \n"
                                    "\n"
                                    "MODEL: ", "MODEL: ")

        self.SELECTED_MODEL = models[modelOption]

        # More options for ResNet
        if self.SELECTED_MODEL == RN:
            option_Resnet = self.askRange(1, len(models),
                                          "Please select specific ResNet Model: \n"
                                          "1. ResNet50 \n"
                                          "2. ResNet101 \n"
                                          "3. ResNet152 \n"
                                          "4. ResNet34 \n"
                                          "5. ResNet18 \n"
                                          "6. ResNet110 \n"
                                          "\n"
                                          "ResNet: ", "ResNet: ")
            self.SELECTED_MODEL = ResNet[option_Resnet]

        # Automatically deals with image size
        self.IMAGE_SIZE = imageSize[modelOption]
        print("Image Size is " + str(self.IMAGE_SIZE))

        if choice == 1:

            print(
                """\n
                +==============================================================+              
                |                   HYPER-PARAMETER SETTINGS                   |                  
                +==============================================================+
                \n""")

            # Hyper-Parameter settings
            if not self.askYesNo("Do you want to use the default values? "):
                self.EPOCHS = self.askDigit("Number of Epoch: ")
                self.BATCH_SIZE = self.askDigit("Batch size: ")
                self.LR = self.askFloat("Learning rate: ")
                self.MOMENTUM = self.askFloat("Momentum: ")
                self.WEIGHT_DECAY = self.askFloat("Weight Decay: ")
                self.FACTOR = self.askFloat("Factor: ")



            print("")

            print(
                """\n
                +==============================================================+              
                |                 MODEL AND DATA SAVE LOCATION                 |                  
                +==============================================================+
                \n""")

            # Splits the file to its training, validation and testing sets
            if len(os.listdir(self.SAVE_LOCATION)) == 0:
                print("Folder still needs to be sorted")
                self.DATA_REBUILD = True
            else:
                print("Folder already sorted")

            # Asks user for their model's name
            self.MODEL_NAME = str(input("Name of saved model: "))

        elif choice == 2:

            print(
                """\n
                +==============================================================+              
                |                   HYPER-PARAMETER SETTINGS                   |                  
                +==============================================================+
                \n""")

            self.EPOCHS = self.askDigit("Number of Epoch: ")
            self.BATCH_SIZE = self.askDigit("Batch size: ")

            print(
                """\n
                +==============================================================+              
                |                 MODEL AND DATA LOAD LOCATION                 |                  
                +==============================================================+
                \n""")

            # Splits the file to its training, validation and testing sets
            if len(os.listdir(self.SAVE_LOCATION)) == 0:
                print("Folder still needs to be sorted")
                self.DATA_REBUILD = True
            else:
                print("FOLDER ALREADY SORTED")

            # Asks user for their model's name
            self.MODEL_NAME = str(input("Name of saved model: "))
        else:

            print(
                """\n
                +==============================================================+              
                |                 MODEL AND DATA LOAD LOCATION                 |                  
                +==============================================================+
                \n""")

            # Splits the file to its training, validation and testing sets
            if len(os.listdir(self.SAVE_LOCATION)) == 0:
                print("Folder still needs to be sorted")
                self.DATA_REBUILD = True
            else:
                print("FOLDER ALREADY SORTED")

            # Asks user for their model's name
            self.MODEL_NAME = str(input("Name of saved model: "))
