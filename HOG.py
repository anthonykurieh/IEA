import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from readCSV import Dict
import numpy as np

def HOG():

    counter = 0

    for dir in os.listdir("resized_images/"):

        img = imread(f"resized_images//{dir}")
        resized = cv2.resize(img,(16,16))
        hog_images = []
        hog_features = []

        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(16, 16),cells_per_block=(4,4),block_norm= 'L2',visualize=True)
        hog_images.append(hog_image)
        hog_features.append(fd)

        counter2 = 0
        for elt in fd:
            Dict[counter]["HOG" + str(counter2)] = int(np.floor(elt*1000))
            counter2 += 1
        counter += 1



   # print(hog_image)

HOG()