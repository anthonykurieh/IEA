import cv2
import numpy as np
import os
from readCSV import Dict
import pandas as pd

counter = 0

for dir in os.listdir("resized_images"):

    img = cv2.imread(f"resized_images//{dir}")
    #img = cv2.imread("resized_images/img001-001.png")
    img2 = img[0:64, 0:64]
   # cv2.imshow("hi",img)
   # cv2.waitKey(0)
    white = np.sum(img == 255)
    black = np.sum(img == 0)
    ratioBW = white / black

    Dict[counter]["ratioBlackWhite"] = ratioBW
    counter += 1
print(Dict)

