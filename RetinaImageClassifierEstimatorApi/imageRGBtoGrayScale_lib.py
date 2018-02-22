import os
import sys
import numpy as np
import matplotlib
import skimage
from skimage import io
from numpy import array
import matplotlib.pyplot as plt
from skimage.viewer import ImageViewer
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import color


def listDataset(path, res1, res2):
    dir_path = path
    # Read the directories available
    list_folders = os.listdir(path)
    labels = []
    images = []

    # Label each folder with a value starting
    # from zero

    label_counter = 0
    for i in list_folders:
        print i + " loading ..." 
        print "The label is " , label_counter
        for k in os.listdir(dir_path+"/"+i+"/"):

                if (k.find("jpg") != "-1" or k.find("jpeg") != "-1"):
                    labels.append(label_counter)
                    
                    #img = Image.open(dir_path+"/"+i+"/"+k)
                    #images.append(array(img))
                    img = io.imread(dir_path+"/"+i+"/"+k)
                    # Resize the image 
                    img_resize = resize(img, (res1,res2))
                    # Convert to gray scale
                    image = color.rgb2gray(img_resize)
                    images.append(image)
        label_counter += 1

    return labels, images



def img2gs_res(image_path,res1,res2):

    img = io.imread(image_path)
    img_resize = resize(img, (res1, res2))
    image = color.rgb2gray(img_resize)
    return image


