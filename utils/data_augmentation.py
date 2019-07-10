import cv2
import numpy as np
from pycocotools.coco import COCO
import random
import os
from imutils import paths

class Augmentation():

    def __init__(self, dataset=None):

        self.dataset = dataset

    def add_hand_to_background(self, background, mask, image, pos_index):

        background[mask==pos_index] = image[mask==pos_index]

        return background

    def create_negative_samples(self):
        
        images_target_path = "/home/alexandre/negative_images/images"
        masks_target_path = "/home/alexandre/negative_images/mask"
        bgs_path = "/home/alexandre/COCO_images"
        masks_path = "/home/alexandre/train_images/mask"
        images_path = "/home/alexandre/train_images/images"
        mixed_masks_path = "/home/alexandre/train_images/mixed_mask"
        bg_paths = sorted(list(paths.list_images(bgs_path)))
        mask_paths = sorted(list(paths.list_images(masks_path)))
    
        for i in range(0, len(mask_paths)):
            mask = cv2.imread(mask_paths[i], 0)
            bg = cv2.imread(bg_paths[i])
            sep = mask_paths[i].split("/")
            sep2 = bg_paths[i].split("/")
            filename = sep[len(sep)-1]
            filename2 = sep2[len(sep2)-1]
            image = cv2.imread(os.path.sep.join([images_path, filename])) 
            mixed_mask = cv2.imread(os.path.sep.join([mixed_masks_path, filename]))

            height, width = mask.shape
            bg = cv2.resize(bg, (width, height))
            #mixed_mask = cv2.resize(mixed_mask, (width, height))
            #image = cv2.resize(image, (width, height))
            #mask = cv2.resize(mask, (width, height))

            pos_index = 0
            background = self.add_hand_to_background(bg, mixed_mask, image, pos_index)
            
            cv2.imwrite(os.path.sep.join([images_target_path, filename2]), background)
            cv2.imwrite(os.path.sep.join([masks_target_path, filename2]), mask)
            

augmentation = Augmentation()
augmentation.create_negative_samples()

"""
dataDir='/home/alexandre/Documentos/TESE/cocoapi'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

coco = COCO(annFile)
superCatIds = coco.getCatIds(supNms=['indoor'])
print(superCatIds)
imgIds = coco.getImgIds(catIds=superCatIds) #had to make a change to the coco.py file
print(str(len(imgIds)))
random.shuffle(imgIds)
coco.download("/home/alexandre/COCO_images", imgIds[:1000])
"""