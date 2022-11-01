import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from readCSV import Dict
import pandas as pd
import csv

Dict2 = []
with open("exec.csv", "r") as data:
    for line in csv.DictReader(data):
        Dict2.append(line)
    print(Dict2.)

path ="testing/0.jpg"

src = cv2.imread(path)

image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
img_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]
img_binary = cv2.bitwise_not(img_thresh)

x1, y1, w, h = cv2.boundingRect(img_binary)
x2 = x1 + w
y2 = y1 + h

start = (x1, y1)
end = (x2, y2)
colour = (255, 255,255)
thickness = 2
width = 128
height = 128
dim = (width,height)

rectangle_img = cv2.rectangle(src, start, end, colour, thickness)
cropped = image[y1:y2,x1:x2]
final = cv2.resize(cropped,dim)
cv2.imshow("hi",final)
cv2.waitKey(0)


def projectHistogramE():
    img = final
    counter = 0
    proj = []
    for column in range(128):

        proj.append(np.sum(img[0:128, column] == 0))
        counter2 = 0
        for elt in proj:
            Dict2[counter]["projHist" + str(counter2)] = elt
            counter2 += 1
        counter += 1
    print(Dict2)

projectHistogramE()
