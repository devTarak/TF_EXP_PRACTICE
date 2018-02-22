import os
import sys
import numpy
import matplotlib
import skimage
from skimage import io
from numpy import array
import matplotlib.pyplot as plt
from skimage.viewer import ImageViewer
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import color


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
        print i + " loading ..."
        for k in os.listdir(dir_path+"/"+i+"/"):
                if (k.find("jpg") != "-1" or k.find("jpeg") != "-1"):
                    labels.append(label_counter)
                    #img = Image.open(dir_path+"/"+i+"/"+k)
                    #images.append(array(img))
                    img = io.imread(dir_path+"/"+i+"/"+k)
                    # Resize the image 
                    img_resize = resize(img, (28,28))
                    # Convert to gray scale
                    image = color.rgb2gray(img_resize)
                    images.append(image)
        label_counter += 1

    return labels, images



path = "/disk1/phase2/data/train"

labels , images = listDataset(path)

print "Printing the first image in the list of images  along with label"


print labels[0]


viewer = ImageViewer(images[0])
viewer.show()


