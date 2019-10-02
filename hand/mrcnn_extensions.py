#python libraries
import os
import sys
import cv2
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

#mrcnn libraries
sys.path.append(ROOT_DIR)  # To find local version of the library
from my_mrcnn.config import Config
from my_mrcnn import utils

class HandConfig(Config):

    NAME = "hand"

    GPU_COUNT = 1
	
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 1 #background + hand

    IMAGE_MIN_DIM = 256 #must be divisable by 2 at least 6 times
    IMAGE_MAX_DIM = 256

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    TRAIN_ROIS_PER_IMAGE = 4

    LEARNING_RATE = 0.0001

    STEPS_PER_EPOCH = 872

    VALIDATION_STEPS = 218

    #TRAIN_BN = None

    #MASK_SHAPE = [56, 56]

"""
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    GRADIENT_CLIP_NORM = 5
"""

class HandDataset(utils.Dataset):
    
    def __init__(self, imagePaths, masksPath):
        super(HandDataset, self).__init__()
        self.imagePaths = imagePaths
        self.masksPath = masksPath

    def load_hands(self, indexes, height, width):
        self.add_class("hand", 1, "vizzy")

        for ind in indexes:
            imagePath = self.imagePaths[ind]
            filename = imagePath.split(os.path.sep)[-1]
            self.add_image("hand", image_id=filename, path=imagePath, width=width, height=height)

    def load_image(self, image_id):
        info = self.image_info[image_id]
        p = self.image_info[image_id]["path"]
        image = cv2.imread(p)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)		
        image = cv2.resize(image, (info['width'], info['height']))

        return image

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "hand":
            return info["hand"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        filename = info["id"]

        annotPath = os.path.sep.join([self.masksPath, filename])
        annotMask = cv2.imread(annotPath)
        #print(annotMask.shape)
        annotMask = cv2.split(annotMask)[0]
        annotMask = cv2.resize(annotMask, (info['width'], info['height']))
        annotMask[annotMask > 0] = 255
        annotMask[annotMask == 0] = 1
        annotMask[annotMask == 255] = 0

        #this is semantic segmentation, convert to instance segmentation when required.
        classIDs = np.unique(annotMask)	

        classIDs = np.delete(classIDs, [0])

        masks = np.zeros((annotMask.shape[0], annotMask.shape[1], 1), dtype="uint8")

        for (i, classID) in enumerate(classIDs):
            classMask = np.zeros(annotMask.shape, dtype="uint8")
            classMask[annotMask==classID] = 1

            masks[:, :, i] = classMask

        return (masks.astype("bool"), classIDs.astype("int32"))
