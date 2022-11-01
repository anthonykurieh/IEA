import cv2
import numpy as np
import os
from readCSV import Dict

counter = 0

#for dir in os.listdir("resized_images/"):

img = cv2.imread("resized_images/img001-001.png")
diagonalSymmetry14 = False

imgT = img[0:64, 64:128]
imgB = img[64:128, 0:64]
cv2.imshow("hi",imgB)
cv2.waitKey(0)

blackT = np.sum(imgT == 0)

blackB = np.sum(imgB == 0)

blackS = blackT / blackB



