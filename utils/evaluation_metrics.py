import numpy as np


class DatasetMasks():

    def __init__(self):

        self.images_masks = []    

    def add_image_masks(self, image_id, gt_masks, pred_masks):
        """Adds the image_id, gt masks and pred masks of a given image, to a list of image
        related data.

        Inputs:
        image_id: the image id, can be the name of the image file 
        (with the respective extension), or an integer, or any other unique id.
        gt_masks: [height, width, N], where N is the number of gt classes in the image 
        (excluding bg).
        pred_masks: [height, width, N], where N is the number of gt classes in the image 
        (excluding bg).
        """

        self.images_masks.append({
            "image_id": image_id,
            "gt_masks": gt_masks,
            "pred_masks": pred_masks
        })    

    def find_image_masks(self, image_id):
        """Uses the image image_id to search the list for the corresponding gt masks and
        pred masks

        Inputs:
        image_id: the image id, can be the name of the image file 
        (with the respective extension), or an integer, or any other unique id.

        Outputs:
        i: {'image_id': string, 
            'gt_masks': [height, width, N], 
            'pred_masks': [height, width, N]}, 
        a dictionary containing the id of the image and the corresponding gt and pred
        masks.
        """

        for i in self.images_masks:
            if i['image_id'] == image_id:
                return i


class EvaluationMetrics():

    def __init__(self, dataset_masks):
        """
        dataset_masks: an object belonging to the class DatasetMasks, containing 
        all the semantic masks of the dataset we which to evaluate, organized in a list of 
        dictionaries.
        """

        self.dataset_masks = dataset_masks

    def iou_mask(self, gt_mask, pred_mask):
        """This function calculates the IoU value of a predicted maks, given its corresponding
        groundthruth mask.

        Inputs: 
        gt_mask: [height, width] binary groundthruth mask.
        pred_mask: [height, width] binary prediction mask.
        
        Returns:
        iou: intersection over union value of the masks
        """

        intersection = np.logical_and(gt_mask, pred_mask)
        union = np.logical_or(gt_mask, pred_mask)
        iou = np.sum(intersection)/np.sum(union)

        return iou

    def compute_avg_iou(self):

        total_iou = 0
        for image_masks in self.dataset_masks.images_masks:
            gt_masks = np.atleast_3d(image_masks["gt_masks"])
            pred_masks = np.atleast_3d(image_masks["pred_masks"])
            
            num_classes = gt_masks.shape[2]
            iou = 0
            for i in range(gt_masks.shape[2]):
                #compute IoU for this class
                print(gt_masks.shape)
                print(pred_masks.shape)
                iou += self.iou_mask(gt_masks[:, :, i], pred_masks[:, :, i])

            #average the IoU over all classes and accumulate in total_iou
            total_iou += float(iou)/float(num_classes)

        #average the IoU over all images and return its value
        print(len(self.dataset_masks.images_masks))
        avg_iou = total_iou/float(len(self.dataset_masks.images_masks))
        return avg_iou