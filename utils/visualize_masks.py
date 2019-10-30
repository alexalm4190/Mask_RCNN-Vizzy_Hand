import cv2
import argparse
from imutils import paths
import os
import sys
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

#mrcnn libraries
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import visualize

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--imagesDir", 
					help='directory where the images and masks are stored',
					required=True)						
arguments = parser.parse_args()

color = np.array((30, 50, 180))
alpha = 0.5

imageNames = os.listdir(os.path.sep.join([arguments.imagesDir, "images"]))
for filename in imageNames:
    image = cv2.imread(os.path.sep.join([arguments.imagesDir, "images", filename]))
    mask = cv2.imread(os.path.sep.join([arguments.imagesDir, "mask", filename]))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    for c in range(3):
        image[:, :, c] = np.where(mask == 0, image[:, :, c] * (1 - alpha)
                         + alpha * color[c] * 255, image[:, :, c])

    cv2.imshow("image", image)      

    cv2.waitKey()               
