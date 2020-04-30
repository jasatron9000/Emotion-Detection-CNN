import os
from PIL import Image as im
from tqdm import tqdm
import numpy as np
import torchvision.transforms.functional as transforms
import random
import torch


def randomCrop(img, width, height):
    prevWidth, prevHeight = img.size

    x = random.randint(0, prevWidth - width)
    y = random.randint(0, prevHeight - height)

    # print(x, y, width, height)
    cropped = transforms.crop(img, y, x, width, height)
    return cropped

    # RandomCrop -> This method randomly crops every image 'amount' times and outputs them as a new


class emotions:
    #  a counter for each class to see how many images has been processed
    afraid_count = 0
    angry_count = 0
    disgust_count = 0
    happy_count = 0
    neutral_count = 0
    sad_count = 0
    surprised_count = 0

    def __init__(self, imgSize):
        self.imgSize = imgSize
        self.training_data = []

    def make_training_data(self, imgLocation: str, colour=True):
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
                    # width, height = image.size

                    # left = (width - width) / 2
                    # top = (height - width) / 2
                    # right = (width + width) / 2
                    # bottom = (height + width) / 2

                    # image = image.crop((left, top, right, bottom))
                    # image = image.resize((self.imgSize, self.imgSize))
                    # Inserts the photo
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

        np.random.shuffle(self.training_data)
        print("Afraid:", self.afraid_count)
        print("Angry:", self.angry_count)
        print("Disgust:", self.disgust_count)
        print("Happy:", self.happy_count)
        print("Neutral:", self.neutral_count)
        print("Sad:", self.sad_count)
        print("Surprised:", self.surprised_count)

    # Custom Transformations for our DataSet
    def save(self, path, percent):
        # creates the
        folders = ["train", "test", "validate"]
        emotion = ["afraid", "angry", "disgust", "happy", "neutral", "sad", "surprised"]

        # make folder
        for i in range(len(folders)):
            folderPath = os.path.join(path, folders[i])
            os.mkdir(folderPath)

            for j in emotion:
                os.mkdir(os.path.join(path, folders[i], j))

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
                image.save(os.path.join(path, "train", emotion[index]) + "\\" + str(i) + ".JPEG")

            for i, validX in tqdm(enumerate(validSet),
                                  desc="Saving training data for \"" + emotion[index] + "\" class",
                                  total=len(trainSet)):
                image = im.fromarray(validX[0])
                image.save(os.path.join(path, "validate", emotion[index]) + "\\" + str(i) + ".JPEG")

            for i, testX in tqdm(enumerate(testSet),
                                 desc="Saving training data for \"" + emotion[index] + "\" class",
                                 total=len(trainSet)):
                image = im.fromarray(testX[0])
                image.save(os.path.join(path, "test", emotion[index]) + "\\" + str(i) + ".JPEG")

            print('\nDone. Completed Saving Photos for ' + emotion[index])

    # list of the flipped image
    def RandomCropData(self, amount: int, width: int, height: int):
        newTrainData = []

        # Iterates through the training data and appends 'amount' images which are cropped images with 'width' and
        # 'height' as the new dimensions of the images
        for i in tqdm(range(len(self.training_data)), desc="Image Process: Randomly Cropping All Images "):  # 4900
            crops = []
            count = amount
            while count > 0:
                convertedImage = randomCrop(self.training_data[i][0], width, height)

                if convertedImage not in crops:
                    crops.append([convertedImage, self.training_data[i][1]])
                else:
                    count += 1

                count -= 1
            # print(len(crops))
            for c in range(len(crops)):
                newTrainData.append(crops[c])

        print("\nFinished Processing: Array Length: " + str(len(newTrainData)))
        self.training_data = newTrainData

    # Image Flip -> This method allows the user to get all the PIL images and flip them and outputs them as a new
    # list of the flipped image
    def ImageFlip(self):
        newTrainData = []

        # Iterates through the training data and appends two images which are the flipped and non-flipped image to
        # the new array
        for i in tqdm(range(len(self.training_data)), desc="Image Process: Flipping All Images"):
            convertedImage = transforms.hflip(self.training_data[i][0])

            newTrainData.append([convertedImage, self.training_data[i][1]])
            newTrainData.append([self.training_data[i][0], self.training_data[i][1]])

        print("\nFinished Processing: Array Length: " + str(len(newTrainData)))

        self.training_data = newTrainData

    # TODO: Implement ChangeToTensor changes the list of PIL Images to a tensor

    # ChangeToTensor -> This method allows you to change the PIL image format to
    def OutputAsTensors(self, device):
        np.random.shuffle(self.training_data)
        X = []
        y = []

        for item in tqdm(self.training_data, desc="Converting to Tensor"):
            X.append(np.array(item[0]))
            y.append(item[1])

        return X, y
