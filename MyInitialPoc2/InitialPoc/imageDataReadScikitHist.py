import os
import sys
import numpy
import matplotlib
import skimage
from skimage import io
from numpy import array
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from pylab import *
from skimage.viewer import ImageViewer
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import color

# mention the path of directory in which the images are
path = ""


def listDataset(path):
    dir_path = path

    # Read the directories available
    list_folders = os.listdir(path)
    labels = []
    images = []

    # Label each folder with a value starting
    # from zero

    label_counter = 0
    for i in list_folders:
        print(i + " loading ...")
        for k in os.listdir(dir_path+"/"+i+"/"):
            if (k.find("jpg") != "-1" or k.find("jpeg") != "-1"):
                labels.append(label_counter)
                img = io.imread(dir_path+"/"+i+"/"+k)
                images.append(img)
        label_counter += 1

    return labels, images




labels , images = listDataset(path)



x1 = [1]
y1 = [labels.count(0)]

x2 = [2]
y2 = [labels.count(1)]

plt.grid(True)
# NoCat grid will be red color
plt.bar(x1,y1,label='NoCat',color='r')

# Cat grid will be green color
plt.bar(x2,y2,label='Cat',color='g')
plt.xlabel("Nocat Or Cat")
plt.ylabel("No of images")
plt.title("Histogram of Cat & Nocat")
plt.xticks((1,2),["NoCat","Cat"])
plt.legend()
plt.show()


