# IMAGE PROCESSING
# =================================================================================================================
# This file contains functions functions that allow the user process a set of custom image dataset to be loaded
# into the CNN for training
# =================================================================================================================

# Imports needed for the following code to work
import os
from os import path
from PIL import Image as im
from tqdm import tqdm
import numpy as np
import random

# emotions -> A class that holds necessary functions and dataset that will be used to train the specifide
#             Also initializes a list which will hold the processed dataset
class emotions:
    #  a counter for each class to see how many images has been processed
    afraid_count = 0
    angry_count = 0
    disgust_count = 0
    happy_count = 0
    neutral_count = 0
    sad_count = 0
    surprised_count = 0

    def __init__(self):
        self.training_data = []

    # make_training_data -> A function that will go through all the image within ready sorted folders and append them
    #                       to a list for augmentation and seperation later on
    # Params:
    #   -imgLocation -> string input: path location of where the sorted pictures are
    def make_training_data(self, imgLocation: str):
# ---------------------------------------- Detect input errors ----------------------------------------------------#
        if (not path.exists(imgLocation)) or (not isinstance(imgLocation, str)):
            raise Exception("Yikes! " + "[" + imgLocation + "]" + " Path does not exist :(")
# -----------------------------------------------------------------------------------------------------------------#



        random.seed(9001)

        # the location of the image files corresponding to class
        afraid = imgLocation + "/afraid"
        angry = imgLocation + "/angry"
        disgust = imgLocation + "/disgust"
        happy = imgLocation + "/happy"
        neutral = imgLocation + "/neutral"
        sad = imgLocation + "/sad"
        surprised = imgLocation + "/surprised"

        labels = {afraid: 0, angry: 1, disgust: 2, happy: 3, neutral: 4, sad: 5, surprised: 6}

        # iterates and processes through all the images from all the classes
        for emotion in labels:
            for f in tqdm(os.listdir(emotion), desc=emotion):
                try:
                    path = emotion + "/" + f
                    image = im.open(path)
                    self.training_data.append([np.array(image), labels[emotion]])
                    image.close()
                    if emotion == afraid:
                        self.afraid_count += 1
                    elif emotion == angry:
                        self.angry_count += 1
                    elif emotion == disgust:
                        self.disgust_count += 1
                    elif emotion == happy:
                        self.happy_count += 1
                    elif emotion == neutral:
                        self.neutral_count += 1
                    elif emotion == sad:
                        self.sad_count += 1
                    elif emotion == surprised:
                        self.surprised_count += 1
                except Exception as e:
                    pass
                    print(str(e))

        # Randomly shuffles the data set so that the classes are not grouped together
        np.random.shuffle(self.training_data)
        print("Afraid:", self.afraid_count)
        print("Angry:", self.angry_count)
        print("Disgust:", self.disgust_count)
        print("Happy:", self.happy_count)
        print("Neutral:", self.neutral_count)
        print("Sad:", self.sad_count)
        print("Surprised:", self.surprised_count)


    # save -> A function that goes through all the images in the processed image list into 3 groups(train, test, valid)
    # and save them into new folders that are also sorted(exactly like original dataset folder)
    # Params:
    #   -Path     -> string input: path location of where the split datasets will be saved to
    #   -percent  -> float input: The percentage of the dataset that will be used for training
    def save(self, Path, percent):
# ---------------------------------------- Detect input errors ----------------------------------------------------#
        if (not path.exists(Path)) or (not isinstance(Path, str)):
            raise Exception("Yikes! " + "[" + Path + "]" + " Path does not exist :(")
        if (not isinstance(percent, float)) or percent > 1:
            raise Exception("Invalid input for percentage, please check that ",
                            percent,
                            " is a decimal percentage thats less than 1")
# -----------------------------------------------------------------------------------------------------------------#


        # creates the
        folders = ["train", "test", "validate"]
        emotion = ["afraid", "angry", "disgust", "happy", "neutral", "sad", "surprised"]

        # make folder
        for i in range(len(folders)):
            folderPath = os.path.join(path, folders[i])
            os.mkdir(folderPath)

            for j in emotion:
                os.mkdir(os.path.join(Path, folders[i], j))

        # Loop through each emotion:
        sortedEmotion = [[], [], [], [], [], [], []]

        # Loop through all emotion
        for data in tqdm(self.training_data, desc="Sorting through the training data"):
            sortedEmotion[data[1]].append(data)

        for index, emote in enumerate(sortedEmotion):
            maxVal = len(emote)
            trainSplit = int(maxVal * percent)
            validSplit = int(maxVal * (percent + ((1 - percent) / 2)))

            trainSet = emote[:trainSplit]
            validSet = emote[trainSplit:validSplit]
            testSet = emote[validSplit:]

            print("\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
            print('Total Images (train, valid, test): {0}/{1}/{2}'.format(str(len(trainSet)),
                                                                          str(len(validSet)),
                                                                          str(len(testSet))))
            print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

            for i, trainX in tqdm(enumerate(trainSet),
                                  desc="Saving training data for \"" + emotion[index] + "\" class",
                                  total=len(trainSet)):
                image = im.fromarray(trainX[0])
                image.save(os.path.join(Path, "train", emotion[index]) + "\\" + str(i) + ".JPEG")

            for i, validX in tqdm(enumerate(validSet),
                                  desc="Saving training data for \"" + emotion[index] + "\" class",
                                  total=len(trainSet)):
                image = im.fromarray(validX[0])
                image.save(os.path.join(Path, "validate", emotion[index]) + "\\" + str(i) + ".JPEG")

            for i, testX in tqdm(enumerate(testSet),
                                 desc="Saving training data for \"" + emotion[index] + "\" class",
                                 total=len(trainSet)):
                image = im.fromarray(testX[0])
                image.save(os.path.join(Path, "test", emotion[index]) + "\\" + str(i) + ".JPEG")

            print('\nDone. Completed Saving Photos for ' + emotion[index])
