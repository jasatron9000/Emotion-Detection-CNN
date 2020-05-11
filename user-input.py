# imports
import sys
import os

sys.path.insert(1, 'models')
# ===================================== Import Architectures =====================================

# Imports needed for the following code to work
import AlexNet as AN
import NishNet as NN
import VGG16 as VGG
import ResNet as RN
import LeNet as LN


# class


class userInput:
    def __init__(self):
        # Store values that are used in the main.py
        # File names
        self.SAVE = False
        self.DATA_REBUILD = False
        self.DATA_LOCATION = ""
        self.LOAD_LOCATION = ""
        self.SAVE_LOCATION = ""
        self.MODEL_SAVE = ""
        self.MODEL_NAME = ""

        # Hyper-Parameter Settings
        self.MOMENTUM = -1
        self.WEIGHT_DECAY = -1
        self.LR = -1
        self.BATCH_SIZE = -1
        self.EPOCHS = -1
        self.IMAGE_SIZE = -1
        self.FACTOR = -1
        self.TRAIN_PERCENT = 0.7

        # Model
        self.SELECTED_MODEL = None

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

    # Function that deals with conflicting types
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

    # Function taht detects if the list of answers are outputed
    @staticmethod
    def askYesNo(question: str):
        answer = input(question)

        while answer != "y" and answer != "n":
            print("Invalid input, please enter either 'y' or 'n' \n")
            answer = input("(y/n): ")

        return answer == "y"

    # Function that detects if a file exists
    @staticmethod
    def askDirectory(questionDirectory: str):
        while True:
            directory = str(input(questionDirectory))

            if not os.path.exists(directory):
                print("Invalid input, this directory does not exist\n")
                continue
            else:
                return directory

    def initValues(self):
        # A
        models = {
            1: LN.LeNet(),
            2: NN.NishNet(),
            3: VGG.CustomVGG13(),
            4: RN
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

        choice = self.askRange(1, 3, "Select which tasks you would like to choose: \n"
                                     "1. Train new network \n"
                                     "2. Train pre-existing network \n"
                                     "3. Evaluate pre-existing network \n"
                                     "\n"
                                     "CHOICE: ")
        if choice == 1:

            # Choose a network to
            modelOption = self.askRange(1, len(models),
                                        "Please select Architecture to use, enter a number to select the model: \n"
                                        "1. LeNet-5 \n"
                                        "2. VGG11 \n"
                                        "3. VGG13 \n"
                                        "4. ResNet \n"
                                        "\n"
                                        "MODEL: ", "MODEL: ")

            self.SELECTED_MODEL = models[modelOption]

            # Detect for ResNet
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

            print(
                """\n
                +==============================================================+              
                |                   HYPER-PARAMETER SETTINGS                   |                  
                +==============================================================+
                \n""")

            self.EPOCHS = self.askDigit("Number of Epoch: ")
            self.BATCH_SIZE = self.askDigit("Batch size: ")
            self.IMAGE_SIZE = self.askDigit("Image size, please input a single integer as the image will be a square: ")
            self.LR = self.askFloat("Learning rate: ")
            self.MOMENTUM = self.askFloat("Momentum: ")
            self.WEIGHT_DECAY = self.askFloat("Weight Decay: ")
            self.FACTOR = self.askFloat("Factor: ")

            # ask for data rebuild
            print("")
            self.DATA_REBUILD = self.askYesNo("Do you want to split and rebuild the data? (y/n)")

            print(
                """\n
                +==============================================================+              
                |                 MODEL AND DATA SAVE LOCATION                 |                  
                +==============================================================+
                \n""")

            if self.DATA_REBUILD:
                self.DATA_LOCATION = self.askDirectory("Path to folder that contains all the sorted location folders: ")
                self.SAVE_LOCATION = self.askDirectory("Path to location to store the split data: ")
            else:
                self.SAVE_LOCATION = self.askDirectory("Path to location to store the split data: ")

            self.MODEL_SAVE = self.askDirectory("Path to location where the trained model will be saved to: ")
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

            self.DATA_REBUILD = self.askYesNo("Do you want to split and rebuild the data? (y/n)")

            if self.DATA_REBUILD:
                self.DATA_LOCATION = self.askDirectory("Path to folder that contains all the sorted location folders: ")
                self.SAVE_LOCATION = self.askDirectory("Path to location to store the split data: ")
            else:
                self.SAVE_LOCATION = self.askDirectory("Path to location to store the split data: ")

            self.MODEL_SAVE = self.askDirectory("Path to location where the trained model will be saved to: ")
            self.MODEL_NAME = str(input("Name of saved model: "))
        else:
            print(
                """\n
                +==============================================================+              
                |                 MODEL AND DATA LOAD LOCATION                 |                  
                +==============================================================+
                \n""")

            self.MODEL_SAVE = self.askDirectory("Path to location where the trained model will be saved to: ")
            self.MODEL_NAME = str(input("Name of saved model: "))

        # ========================================================================#
        # Then ask to choose a network to train
        #   If Resnet is the user can choose a particular network
        # If they want to train the network ask if they want to rebuild data (y/n)
        # Ask for the hyper-parameters
        #
        # ========================================================================#
        # If want to get an existing network ask for them to enter the fileLocation
        # Keep asking until it that files exists
        #
        pass


UI = userInput()

UI.initValues()
