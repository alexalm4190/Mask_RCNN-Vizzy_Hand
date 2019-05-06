import cv2
import argparse
import os
import numpy as np
from skimage import draw

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--imagesDir", 
					help='directory where the real images are stored',
					required=True)						
arguments = parser.parse_args()

rowCoords = []
columnCoords = []
clone = []
image = []

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
    global refPt
    global clone
    global image
	# if the left mouse button was clicked, record the (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        rowCoords.append(y)
        columnCoords.append(x)
        if len(rowCoords) > 1:
            clone = image.copy()
            cv2.line(image, (x, y), (columnCoords[len(columnCoords)-2], rowCoords[len(rowCoords)-2]), 100)    
            cv2.imshow("image", image)

    elif event == cv2.EVENT_MBUTTONDOWN:
        rowCoords.pop()
        columnCoords.pop()
        image = clone.copy()

imageNames = os.listdir(os.path.sep.join([arguments.imagesDir, "images"]))
for filename in imageNames:
    image = cv2.imread(os.path.sep.join([arguments.imagesDir, "images", filename]))
    image = cv2.resize(image,(1000, 600), interpolation=cv2.INTER_LINEAR)
    mask = np.zeros((image.shape[0], image.shape[1]))
    mask[:] = 255

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, break from the loop
        if key == ord("q"):
            break        

    #refPt.append(refPt[0])
    
    #mask[:] = tuple((255, 255, 255))
    #cv2.fillConvexPoly(mask, np.array(refPt), tuple((0, 0, 0)))
    fill_row_coords, fill_col_coords = draw.polygon(np.array(rowCoords), np.array(columnCoords), image.shape)
    mask[fill_row_coords, fill_col_coords] = 0
    cv2.imwrite(os.path.sep.join([arguments.imagesDir, "mask", filename]), mask)
    
    rowCoords = []
    columnCoords = []