import os
import sys
import numpy
from PIL import Image
from numpy import array

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
        for j in os.listdir(dir_path+"/"+i):
            for k in os.listdir(dir_path+"/"+i+"/"):
                labels.append(label_counter)
                img = Image.open(dir_path+"/"+i+"/"+k)
                images.append(array(img))
        label_counter += 1

    return labels, images



path = "/home/biarca/disk1/phase2/Tarak/stare-tensorflow-estimator/stare/train"

labels , images = listDataset(path)

print "Printing the top 5 images in the list of images  along with labels"


counter = 0
for i in images:
   print labels[counter]
   print i
   counter += 1
   if counter > 5:
      break
