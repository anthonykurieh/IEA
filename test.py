import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

image_path = 'archive//Img'
width = 128
height = 128
dim = (width,height)

for x in os.listdir(image_path):
    #print(x)
    cropped_path = "cropped_images/"+x
    resized_path = "resized_images/"+x
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
    cv2.imwrite(cropped_path,cropped)
    cv2.imwrite(resized_path,resize)


