# Code that allows you to convert the FER+ CVS file to the seperate pictures
import csv
import numpy as np
from PIL import Image as img
import os
import shutil
from tqdm import tqdm


# ========================================= KDEF DATASET ================================================
# Ask user if they want to use the KDEF dataset or not
use_KDEF = input("Use KDEF dataset? (y/n): ")

# ========================================= Detect input errors ================================================
if use_KDEF == "y":

    # Input address to the address with the source folder where the KDEF images are kept
    src = input("Enter path to where FDEF is: ")

    if not os.path.exists(src):
        raise Exception("Directory {} does not exist".format(src))

    if not os.path.exists(src + "/KDEF_sorted"):
        os.mkdir(src + "/KDEF_sorted")
# ==============================================================================================================

    # Initializes the folders
    dst_afraid = src + "/KDEF_sorted/afraid"
    dst_angry = src + "/KDEF_sorted/angry"
    dst_disgust = src + "/KDEF_sorted/disgust"
    dst_happy = src + "/KDEF_sorted/happy"
    dst_neutral = src + "/KDEF_sorted/neutral"
    dst_sad = src + "/KDEF_sorted/sad"
    dst_surprised = src + "/KDEF_sorted/surprised"

    # count for number of images that has be organised into the correct folder
    count = 0

    # List of paths to the emotion folders
    sorted_folders = [dst_afraid, dst_angry, dst_disgust, dst_happy, dst_neutral, dst_sad, dst_surprised]

    # Change the source path to where the images are held for the KDEF dataset
    src = src + "\KDEF_and_AKDEF\KDEF"
    os.chdir(src)

    # Create folders if there aren't any in there already
    for i in sorted_folders:
        if not os.path.exists(i):
            os.mkdir(i)
        else:
            raise Exception("Directory {} already contains a folder {}".format(src, i))

    # loop through all the pictures in the KDEF dataset and organise into the the correct folders based on image label
    for f in tqdm(os.listdir(), desc="Organising KDEF dataset"):

        fileName, fileExt = os.path.splitext(f)
        os.chdir(src + "/" + fileName)

        for i in os.listdir():
            try:
                imgName, imgExt = os.path.splitext(i)
                emotion = imgName[4] + imgName[5]

                if (emotion == 'AF'):
                    shutil.copy(src=src + "/" + f + "/" + i, dst=dst_afraid)
                elif (emotion == 'AN'):
                    shutil.copy(src=src + "/" + f + "/" + i, dst=dst_angry)
                elif (emotion == 'DI'):
                    shutil.copy(src=src + "/" + f + "/" + i, dst=dst_disgust)
                elif (emotion == 'HA'):
                    shutil.copy(src=src + "/" + f + "/" + i, dst=dst_happy)
                elif (emotion == 'NE'):
                    shutil.copy(src=src + "/" + f + "/" + i, dst=dst_neutral)
                elif (emotion == 'SA'):
                    shutil.copy(src=src + "/" + f + "/" + i, dst=dst_sad)
                elif (emotion == 'SU'):
                    shutil.copy(src=src + "/" + f + "/" + i, dst=dst_surprised)
                else:
                    count += 1
            except Exception as e:
                pass
                print("please check the following for naming errors: {} and re-run".format(str(e)))
elif use_KDEF == "n":
    pass
# ========================================= Detect input errors ================================================
else:
    raise Exception("{} Is not a valid answer, please use 'y' or 'n'".format(use_KDEF))
# ==============================================================================================================


# ========================================= FER DATASET ================================================
file = open("train.csv", "r")
emotion = ["angry", "disgust", "afraid", "happy",  "sad", "surprised", "neutral"]
path = r"data"

os.mkdir(path)
os.mkdir(r"edited")
os.mkdir(r"saved")

for j in emotion:
     os.mkdir(os.path.join(path, j))

try:
    csv_reader = csv.reader(file)
    count = 0
    for line in tqdm(csv_reader, desc="Converting CVS to PNG", total=28709):
        if count != 0:
            lengthCount = 0
            imageMatrix = []
            for w in range(48):
                listRow = []

                for h in range(48):
                    s = ""
                    while line[1][lengthCount] != ' ':
                        s += line[1][lengthCount]
                        lengthCount += 1

                        if lengthCount == len(line[1]):
                            break

                    lengthCount += 1
                    listRow.append(int(s))
                imageMatrix.append(listRow)

            imageMatrix = np.uint8(imageMatrix)
            image = img.fromarray(imageMatrix, 'L')

            detectedPath = os.path.join(path, emotion[int(line[0])])
            image.save(detectedPath + "/" + str(count) + ".png")

        count += 1

finally:
    file.close()
