import cv2
import numpy as np
import os
from readCSV import Dict
from projectionHistogram import projectionHistogram,projectionHistogramH
from ratioBlackAndWhite import ratioBW
import pandas as pd
#from HOG import HOG


def verticalSymmetry():
    counter = 0
    for dir in os.listdir("resized_images/"):

        img = cv2.imread(f"resized_images//{dir}")

        verticalSymmetry = 0

        imgL = img[0:128,0:64]
        imgR = img[0:128,64:128]

        blackL = np.sum(imgL == 0)

        blackR = np.sum(imgR == 0)

        blackS = blackR/blackL



        if 0.85<=blackS<=1.25:
            verticalSymmetry = 1
        Dict[counter]["verticalSymmetry"] = verticalSymmetry
        counter += 1


def horizontalSymmetry():
    counter = 0

    for dir in os.listdir("resized_images/"):

        img = cv2.imread(f"resized_images//{dir}")

        horizontalSymmetry = 0

        imgT = img[0:64, 0:128]
        imgB = img[64:128, 0:128]

        blackT = np.sum(imgT == 0)

        blackB = np.sum(imgB == 0)

        blackS = blackT / blackB

        if 0.85 <= blackS <= 1.25:
            horizontalSymmetry = 1
        Dict[counter]["horizontalSymmetry"] = horizontalSymmetry
        counter += 1

def diagonalSymmetry14():
    counter = 0

    for dir in os.listdir("resized_images/"):

        img = cv2.imread(f"resized_images//{dir}")

        diagonalSymmetry14 = 0

        imgT = img[0:64, 0:64]
        imgB = img[64:128, 64:128]

        blackT = np.sum(imgT == 0)

        blackB = np.sum(imgB == 0)

        blackS = blackT / blackB

        if 0.85 <= blackS <= 1.25:
            diagonalSymmetry14 = 1
        Dict[counter]["diagonalSymmetry14"] = diagonalSymmetry14
        counter += 1

def diagonalSymmetry23():
    counter = 0

    for dir in os.listdir("resized_images/"):

        img = cv2.imread(f"resized_images//{dir}")

        diagonalSymmetry23 = 0

        imgT = img[0:64, 64:128]
        imgB = img[64:128, 0:64]

        blackT = np.sum(imgT == 0)
        blackB = np.sum(imgB == 0)

        blackS = blackT / (blackB+0.0000001)

        if 0.85 <= blackS <= 1.25:
            diagonalSymmetry23 = 1
        Dict[counter]["diagonalSymmetry23"] = diagonalSymmetry23
        counter += 1

if __name__ == '__main__':
    ratioBW()
    verticalSymmetry()
    horizontalSymmetry()
    diagonalSymmetry14()
    diagonalSymmetry23()
    projectionHistogram()
    projectionHistogramH()
    #HOG()

    df = pd.DataFrame(Dict)
    df.to_csv("features.csv", )
    print(Dict)