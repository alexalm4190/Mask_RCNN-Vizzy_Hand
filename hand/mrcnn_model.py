#python libraries
import os
import sys
import numpy as np
import tensorflow

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

#mrcnn libraries
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
from mrcnn import visualize

class Model():

    def __init__(self, modelDir, trainedWeightsPath, device, dataset_train, dataset_val, dataset_test=None):     
        self.modelDir = modelDir
        self.trainedWeightsPath = trainedWeightsPath
        self.device = device
        self.dataset_train = dataset_train 
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test

    def train_model(self, config):  
        image_ids = np.random.choice(self.dataset_train.image_ids, 4)
        for image_id in image_ids:
            image = self.dataset_train.load_image(image_id)
            mask, class_ids = self.dataset_train.load_mask(image_id
        )
            visualize.display_top_masks(image, mask, class_ids, self.dataset_train.class_names)

        with tensorflow.device(self.device):
            model = modellib.MaskRCNN(mode="training", config=config, model_dir=self.modelDir)

        init_with = "coco"

        if init_with == "imagenet":
            model.load_weights(model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            model.load_weights(self.trainedWeightsPath, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        elif init_with == "last":
            model.load_weights(model.find_last(), by_name=True)

        model.train(self.dataset_train, self.dataset_val, learning_rate=config.LEARNING_RATE, epochs=50, layers='heads')
        model.train(self.dataset_train, self.dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=100, layers="all")

    def test_model(self, inference_config, modelPath):
        with tensorflow.device(self.device):
            model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=self.modelDir)
        
        print("Loading weights from", modelPath)
        model.load_weights(modelPath, by_name=True)

        #image_id = random.choice(dataset_val.image_ids)
        """
        log("original_image", original_image)
        log("image_meta", image_meta)
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)
        """

        #for image_path in IMAGE_PATHS:
        if self.dataset_test != None:
            for image_id in self.dataset_test.image_ids:
                original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                    modellib.load_image_gt(self.dataset_test, inference_config, image_id, use_mini_mask=False)

                results = model.detect([original_image], verbose=1)
                r = results[0]
                visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], self.dataset_test.class_names, r['scores'], figsize=(8, 8))
        else:
            for image_id in self.dataset_val.image_ids:
                original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                    modellib.load_image_gt(self.dataset_val, inference_config, image_id, use_mini_mask=False)

                visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, self.dataset_val.class_names, figsize=(8, 8))

                results = model.detect([original_image], verbose=1)
                r = results[0]
                visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], self.dataset_val.class_names, r['scores'], figsize=(8, 8))

        # Compute VOC-Style mAP @ IoU=0.5
        # Running on 10 images. Increase for better accuracy.
        """
        image_ids = np.random.choice(self.dataset_val.image_ids, 10)
        APs = []
        arguments.testDataset = False
        for image_id in image_ids:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)
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
        """
