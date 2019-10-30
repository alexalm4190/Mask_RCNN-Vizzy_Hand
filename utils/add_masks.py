import argparse
import os
import sys
import cv2
from imutils import paths
import numpy as np

#renames images and adds blank masks

# Initialize the parser
parser = argparse.ArgumentParser()
# Add arguments to the parser
parser.add_argument('-i', "--imagesDir", 
					help='choose the directory where the images are stored',
					required=True)
parser.add_argument('-m', "--masksDir", 
					help='choose the directory in which to save the blank masks',
					required=True)
parser.add_argument('-n', "--number", 
					help='number to start enumerating the images from (number of already existing positive images)',
					required=True)						
# Parse the arguments
arguments = parser.parse_args()

imageNames = os.listdir(arguments.imagesDir)
n = int(arguments.number)
for filename in imageNames:
    extension = filename.split('.')[1]
    image = cv2.imread(os.path.sep.join([arguments.imagesDir, filename]))

    zeros = ""
    for i in range(0, 4-len(arguments.number)):
        zeros += '0'
    oldFile = os.path.join(arguments.imagesDir, filename)
    newName = "FrameBuffer_" + zeros + str(n) + '.' + extension
    newFile = os.path.join(arguments.imagesDir, newName)
    os.rename(oldFile, newFile)
    n += 1
    
    image = np.full(image.shape, 255, np.uint8) 
    cv2.imwrite(os.path.sep.join([arguments.masksDir, newName]), image)
