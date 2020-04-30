# Code that allows you to convert the FER+ CVS file to the seperate pictures
import csv
import numpy as np
from PIL import Image as img
import os
from tqdm import tqdm

file = open("train.csv", "r")
emotion = ["angry", "disgust", "afraid", "happy",  "sad", "surprised", "neutral"]
path = r"D:\Biggie Cheese\Desktop\Uni\302\CS302-Python-2020-Group25\data"
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
