import os
import random

path = "numbers/"
path2 = "numbers/0"
counter = 0


for x in os.listdir(path):
    data = []
    for y in os.listdir(path+x):
        data.append(y)
    random.shuffle(data)

    print(data)
