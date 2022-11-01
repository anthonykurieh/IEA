import cv2
import numpy as np
import os
from readCSV import Dict
import pandas as pd

def ratioBW():
  counter = 0


  for dir in os.listdir("resized_images/"):
    #  print(dir)
      img = cv2.imread(f"resized_images//{dir}")
      white = np.sum(img == 255)
      black = np.sum(img == 0)
      ratioBW = white/black

      Dict[counter]["ratioBlackWhite"] = ratioBW
      counter += 1



#print(Dict)

#df.items()
