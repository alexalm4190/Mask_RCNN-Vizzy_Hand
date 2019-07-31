import argparse
import cv2
import os
from imutils import paths

# Initialize the parser
parser = argparse.ArgumentParser()
# Add arguments to the parser
parser.add_argument('-n', "--namesDir", 
					help='choose the directory from where you want select the filenames',
					required=True)
parser.add_argument('-s', "--sourceDir", 
					help='choose the directory from where you want to copy the images',
					required=True)
parser.add_argument('-t', "--targetDir", 
					help='choose the directory to where you want to paste the images',
					required=True)     
# Parse the arguments
arguments = parser.parse_args()

filenames = os.listdir(arguments.namesDir)

for filename in filenames:
    sourcePath = os.path.sep.join([arguments.sourceDir, filename])
    image = cv2.imread(sourcePath)

    targetPath = os.path.sep.join([arguments.targetDir, filename])
    cv2.imwrite(targetPath, image)

