import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import csv
import pandas as pd
from Model import neigh,dt,model

image_path = 'testing'
width = 128
height = 128
dim = (width,height)

Dict2 = []
with open("exec.csv", "r") as data:
    for line in csv.DictReader(data):
        Dict2.append(line)

for x in os.listdir(image_path):
    #print(x)
    #cropped_path = "cropped_images/"+x
    resized_path = "testing_resized/"+x
    src = cv2.imread(image_path+'//'+x)

    image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    img_thresh =cv2.threshold(image,0,255,cv2.THRESH_BINARY)[1]
    img_binary = cv2.bitwise_not(img_thresh)


    x1,y1,w,h = cv2.boundingRect(img_binary)
    x2 = x1+w
    y2 = y1+h

    start = (x1, y1)
    end = (x2, y2)
    colour = (255, 255,255)
    thickness = 2

    rectangle_img = cv2.rectangle(src, start, end, colour, thickness)
   # print("x1:", x1, "x2:", x2, "y1:", y1, "y2:", y2)
    cropped = image[y1:y2,x1:x2]
    resize = cv2.resize(cropped,dim)
    #cv2.imwrite(cropped_path,cropped)
    cv2.imwrite(resized_path,resize)

def projHistE():
    counter = 0

    for dir in os.listdir("testing_resized"):

        # img = cv2.imread("resized_images/img001-002.png")
        img = cv2.imread(f"testing_resized//{dir}")
        # cv2.imshow("",img)
        # cv2.waitKey(0)
        proj = []
        for column in range(128):
            proj.append(np.sum(img[0:128, column] == 0))
        counter2 = 0
        for elt in proj:
            Dict2[counter]["projHist" + str(counter2)] = elt
            counter2 += 1
        counter += 1

def projHistHE():
    counter = 0

    for dir in os.listdir("testing_resized/"):

        #img = cv2.imread("resized_images/img001-002.png")
        img = cv2.imread(f"testing_resized//{dir}")
       # cv2.imshow("",img)
        #cv2.waitKey(0)
        proj = []
        for row in range(128):

            proj.append(np.sum(img[row,0:128] == 0))
        counter2 = 0
        for elt in proj:
            Dict2[counter]["projHistH" + str(counter2)] = elt
            counter2 += 1
        counter += 1
def verticalSymmetryE():
    counter = 0
    for dir in os.listdir("testing_resized/"):

        # img = cv2.imread("resized_images/img001-002.png")
        img = cv2.imread(f"testing_resized//{dir}")

        verticalSymmetry = 0

        imgL = img[0:128,0:64]
        imgR = img[0:128,64:128]

        blackL = np.sum(imgL == 0)

        blackR = np.sum(imgR == 0)

        blackS = blackR/blackL



        if 0.85<=blackS<=1.25:
            verticalSymmetry = 1
        Dict2[counter]["verticalSymmetry"] = verticalSymmetry
        counter += 1


def horizontalSymmetryE():
    counter = 0

    for dir in os.listdir("testing_resized/"):

        # img = cv2.imread("resized_images/img001-002.png")
        img = cv2.imread(f"testing_resized//{dir}")

        horizontalSymmetry = 0

        imgT = img[0:64, 0:128]
        imgB = img[64:128, 0:128]

        blackT = np.sum(imgT == 0)

        blackB = np.sum(imgB == 0)

        blackS = blackT / blackB

        if 0.85 <= blackS <= 1.25:
            horizontalSymmetry = 1
        Dict2[counter]["horizontalSymmetry"] = horizontalSymmetry
        counter += 1

def diagonalSymmetry14E():
    counter = 0

    for dir in os.listdir("testing_resized/"):

        # img = cv2.imread("resized_images/img001-002.png")
        img = cv2.imread(f"testing_resized//{dir}")

        diagonalSymmetry14 = 0

        imgT = img[0:64, 0:64]
        imgB = img[64:128, 64:128]

        blackT = np.sum(imgT == 0)

        blackB = np.sum(imgB == 0)

        blackS = blackT / blackB

        if 0.85 <= blackS <= 1.25:
            diagonalSymmetry14 = 1
        Dict2[counter]["diagonalSymmetry14"] = diagonalSymmetry14
        counter += 1

def diagonalSymmetry23E():
    counter = 0

    for dir in os.listdir("testing_resized/"):

        # img = cv2.imread("resized_images/img001-002.png")
        img = cv2.imread(f"testing_resized//{dir}")

        diagonalSymmetry23 = 0

        imgT = img[0:64, 64:128]
        imgB = img[64:128, 0:64]

        blackT = np.sum(imgT == 0)
        blackB = np.sum(imgB == 0)

        blackS = blackT / (blackB+0.0000001)

        if 0.85 <= blackS <= 1.25:
            diagonalSymmetry23 = 1
        Dict2[counter]["diagonalSymmetry23"] = diagonalSymmetry23
        counter += 1

projHistE()
projHistHE()
verticalSymmetryE()
horizontalSymmetryE()
diagonalSymmetry14E()
diagonalSymmetry23E()


#print(Dict2)

df2 = pd.DataFrame(Dict2)
df2.to_csv("features_exec.csv",)

#df2.drop(['counter'], axis=1)

model.predict(df2)
dt.predict(df2)
neigh.predict(df2)
