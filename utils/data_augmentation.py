import cv2
import numpy as np
from pycocotools.coco import COCO
import random

class Augmentation():

    def __init__(self, dataset=None):

        self.dataset = dataset

    def add_hand_to_background(self, background, mask, image, pos_index):

        background[mask==pos_index] = image[mask==pos_index]

        return background

    def create_negative_sample(self):
        
        mask = cv2.imread("/home/alexandre/train_images/mask/FrameBuffer_0000.png", 0)
        image = cv2.imread("/home/alexandre/train_images/images/FrameBuffer_0000.png")
        background = cv2.imread("/home/alexandre/teste.jpeg")
        background = cv2.resize(background, (1280, 960))

        pos_index = 0
        background = self.add_hand_to_background(background, mask, image, pos_index)
        return background

#augmentation = Augmentation()
#image = augmentation.create_negative_sample()

#cv2.imwrite("/home/alexandre/test_result.png", image)

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