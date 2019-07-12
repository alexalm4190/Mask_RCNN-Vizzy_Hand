#python libraries
import os
import sys
import numpy as np
import tensorflow
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

#mrcnn libraries
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn import utils
from utils import evaluation_metrics

class Model():

    def __init__(self, modelDir, device):     
        self.modelDir = modelDir
        self.device = device

    def train_model(self, config, trainedWeightsPath, dataset_train, dataset_val):  
        image_ids = np.random.choice(dataset_train.image_ids, 4)
        for image_id in image_ids:
            image = dataset_train.load_image(image_id)
            mask, class_ids = dataset_train.load_mask(image_id)
            visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
        image_ids = np.random.choice(dataset_val.image_ids, 4)
        for image_id in image_ids:
            image = dataset_val.load_image(image_id)
            mask, class_ids = dataset_val.load_mask(image_id)
            visualize.display_top_masks(image, mask, class_ids, dataset_val.class_names)

        with tensorflow.device(self.device):
            model = modellib.MaskRCNN(mode="training", config=config, model_dir=self.modelDir)

        init_with = "coco"

        if init_with == "imagenet":
            model.load_weights(model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            model.load_weights(trainedWeightsPath, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        elif init_with == "last":
            model.load_weights(model.find_last(), by_name=True)

        model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=20, layers='heads')
        model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=50, layers="all")

    def test_model(self, inference_config, modelPath, dataset_test):
        with tensorflow.device(self.device):
            model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=self.modelDir)
        
        print("Loading weights from", modelPath)
        model.load_weights(modelPath, by_name=True)
 
        """
        for image_id in dataset_test.image_ids:
                
            original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset_test, inference_config, image_id, use_mini_mask=False)

            results = model.detect([original_image], verbose=1)
            r = results[0]
            path = "/home/alexandre/Documentos/TESE/results/experiments/" + str(image_id) + ".png"
            visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dataset_test.class_names, r['scores'], figsize=(8, 8),
                                        save_path=path)
        
        print("computing mAP...")
        # Compute VOC-Style mAP @ IoU=0.5
        APs = []
        for image_id in dataset_test.image_ids:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset_test, inference_config, image_id, use_mini_mask=False)
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
        print("computing average IoU and BDE...")
        dataset_masks = evaluation_metrics.DatasetMasks()
        for image_id in dataset_test.image_ids:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset_test, inference_config, image_id, use_mini_mask=False)
            # Run object detection
            results = model.detect([image], verbose=0)
            r = results[0]
            if r['masks'].shape[2] == 0:
                pred_mask = np.zeros((inference_config.IMAGE_MIN_DIM, inference_config.IMAGE_MIN_DIM, 1), dtype="bool")
            else:   
                pred_mask = r['masks'][:, :, 0]
                for i in range(1, r['masks'].shape[2]):
                    pred_mask = np.logical_or(pred_mask, r['masks'][:, :, i])
            #print(gt_mask.shape)
            #print(r['masks'].shape)
            dataset_masks.add_image_masks(image_id, gt_mask, pred_mask)
        metric = evaluation_metrics.EvaluationMetrics(dataset_masks)
        avg_iou, avg_bde = metric.compute_avg_iou_bde()
        print("avg IoU: ", avg_iou)
        print("avg BDE: ", avg_bde)

"""
        else:
            for image_id in self.dataset_val.image_ids:
                original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                    modellib.load_image_gt(self.dataset_val, inference_config, image_id, use_mini_mask=False)

                visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, self.dataset_val.class_names, figsize=(8, 8))

                results = model.detect([original_image], verbose=1)
                r = results[0]
                visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], self.dataset_val.class_names, r['scores'], figsize=(8, 8))
"""