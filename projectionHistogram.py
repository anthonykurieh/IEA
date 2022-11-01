import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from readCSV import Dict
import pandas as pd



def projectionHistogram():
    counter = 0

    for dir in os.listdir("resized_images/"):

        #img = cv2.imread("resized_images/img001-002.png")
        img = cv2.imread(f"resized_images//{dir}")
       # cv2.imshow("",img)
        #cv2.waitKey(0)
        proj = []
        for column in range(128):

            proj.append(np.sum(img[0:128, column] == 0))
        counter2 = 0
        for elt in proj:
            Dict[counter]["projHist" + str(counter2)] = elt
            counter2 += 1
        counter += 1

def projectionHistogramH():
    counter = 0

    for dir in os.listdir("resized_images/"):

        #img = cv2.imread("resized_images/img001-002.png")
        img = cv2.imread(f"resized_images//{dir}")
       # cv2.imshow("",img)
        #cv2.waitKey(0)
        proj = []
        for row in range(128):

            proj.append(np.sum(img[row,0:128] == 0))
        counter2 = 0
        for elt in proj:
            Dict[counter]["projHistH" + str(counter2)] = elt
            counter2 += 1
        counter += 1


projectionHistogram()
projectionHistogramH()

#df = pd.DataFrame(Dict)
#df.to_csv("features.csv",)