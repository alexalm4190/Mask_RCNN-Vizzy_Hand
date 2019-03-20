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

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
HOME_DIR = os.path.abspath("../../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import imutils
from imutils import paths

IMAGES_PATH = os.path.sep.join([HOME_DIR, "gt_images/images"])
MASKS_PATH = os.path.sep.join([HOME_DIR, "gt_images/mask"])

TRAINING_SPLIT = 0.7
VALIDATION_SPLIT = 0.2

IMAGE_PATHS = sorted(list(paths.list_images(IMAGES_PATH)))
idxs = list(range(0, len(IMAGE_PATHS)))
random.seed(None)
random.shuffle(idxs)
i = int(len(idxs) * TRAINING_SPLIT)
i_val = int(len(idxs) * (VALIDATION_SPLIT + TRAINING_SPLIT))
trainIdxs = idxs[:i]
valIdxs = idxs[i:i_val]
testIdxs = idxs[i_val:]

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
	utils.download_trained_weights(COCO_MODEL_PATH)

class HandConfig(Config):
	NAME = "hand"

	GPU_COUNT = 1
	IMAGES_PER_GPU = 8

	NUM_CLASSES = 1 + 1 #background + hand

	IMAGE_MIN_DIM = 128
	IMAGE_MAX_DIM = 128

	RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

	TRAIN_ROIS_PER_IMAGE = 32

	STEPS_PER_EPOCH = 100

	VALIDATION_STEPS = 5

config = HandConfig()
config.display()

def get_ax(rows=1, cols=1, size=8):

	_, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
	return ax

class HandDataset(utils.Dataset):
	"""
	def load_hands_test(self, indexes, height, width): #in case we want to test the network on other images
		self.add_class("hand", 1, "vizzy")

		for ind in indexes:
			imagePath = TEST_IMAGE_PATHS[ind]
			filename = imagePath.split(os.path.sep)[-1]
			self.add_image("hand", image_id=filename, path=imagePath, width=width, height=height)
	"""
	def load_hands(self, indexes, height, width):
		self.add_class("hand", 1, "vizzy")

		for ind in indexes:
			imagePath = IMAGE_PATHS[ind]
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
		annotPath = os.path.sep.join([MASKS_PATH, filename])

		annotMask = cv2.imread(annotPath)
		annotMask = cv2.split(annotMask)[0]
		annotMask = cv2.resize(annotMask, (info['width'], info['height']))
		annotMask[annotMask > 0] = 255
		annotMask[annotMask == 0] = 1
		annotMask[annotMask == 255] = 0

		classIDs = np.unique(annotMask)	
 
		classIDs = np.delete(classIDs, [0])

		masks = np.zeros((annotMask.shape[0], annotMask.shape[1], 1), dtype="uint8")

		for (i, classID) in enumerate(classIDs):
			classMask = np.zeros(annotMask.shape, dtype="uint8")
			classMask[annotMask==classID] = 1

			masks[:, :, i] = classMask

		return (masks.astype("bool"), classIDs.astype("int32"))

#Training dataset
dataset_train = HandDataset()
dataset_train.load_hands(trainIdxs, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = HandDataset()
dataset_val.load_hands(valIdxs, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

# Test dataset
dataset_test = HandDataset()
dataset_test.load_hands(testIdxs, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_test.prepare()

"""
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
	image = dataset_train.load_image(image_id)
	mask, class_ids = dataset_train.load_mask(image_id
)
	visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

init_with = "coco"

if init_with == "imagenet":
	model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
	model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
	model.load_weights(model.find_last(), by_name=True)

model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=1, layers='heads')

model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=2, layers="all")
"""

class InferenceConfig(HandConfig):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

model_path = model.find_last()

print("Loading weights from", model_path)
model.load_weights(model_path, by_name=True)

image_id = random.choice(dataset_test.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
	modellib.load_image_gt(dataset_test, inference_config, image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

original_image = cv2.imread('/home/alexandre/hand_robot.png')

#visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset_test.class_names, figsize=(8, 8))

results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dataset_test.class_names, r['scores'], figsize=(8, 8))

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_test.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_test, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))
