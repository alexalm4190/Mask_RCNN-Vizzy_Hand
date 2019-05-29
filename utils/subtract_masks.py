import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import argparse
from imutils import paths

# Initialize the parser
parser = argparse.ArgumentParser()
# Add arguments to the parser
parser.add_argument('-i', "--imagesDir", 
					help='choose the directory where the images are stored',
					required=True)
# Parse the arguments
arguments = parser.parse_args()

MASKS_PATH = os.path.sep.join([arguments.imagesDir, "gt_images/mask"])
BG_MASKS_PATH = os.path.sep.join([arguments.imagesDir, "gt_images/bg_mask"])
MASKS_TARGET_PATH = os.path.sep.join([arguments.imagesDir, "train_images/mask"])

#get mask and depth images of the hand
depthMaskFiles = []
maskFiles = []

for i in sorted(list(paths.list_images(MASKS_PATH))):
    if 'Depth' in i:
        depthMaskFiles.append(i)
    elif 'FrameBuffer' in i:
        maskFiles.append(i)    

#get mask and depth images of background objects
depthMaskFiles_bg = []
maskFiles_bg = []

for i in sorted(list(paths.list_images(BG_MASKS_PATH))):
    if 'Depth' in i:
        depthMaskFiles_bg.append(i)
    elif 'FrameBuffer' in i:
        maskFiles_bg.append(i)    

k = 0
for k in range(0, len(maskFiles)):
    img_depthMask = cv2.imread(depthMaskFiles[k])
    img_depthMask = cv2.cvtColor(img_depthMask, cv2.COLOR_BGR2GRAY)

    img_mask = cv2.imread(maskFiles[k])
    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)

    img_depthMask_bg = cv2.imread(depthMaskFiles_bg[k])
    img_depthMask_bg = cv2.cvtColor(img_depthMask_bg, cv2.COLOR_BGR2GRAY)

    #img_mask_bg = cv2.imread(maskFiles_bg[k])
    #img_mask_bg = cv2.cvtColor(img_mask_bg, cv2.COLOR_BGR2GRAY)

    img_mask[img_depthMask > img_depthMask_bg] += 255
    img_mask[img_mask > 255] = 255

    sep_path = maskFiles[k].split(".")
    ext = sep_path[1]
    filename = sep_path[0]
    filename = filename.split("/")
    cv2.imwrite(MASKS_TARGET_PATH + '/' + filename[len(filename)-1] + '.' + ext, img_mask)

