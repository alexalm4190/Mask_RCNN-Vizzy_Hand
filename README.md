# Mask_RCNN-Vizzy_Hand

The purpose of this repository is to use [Mask RCNN](https://arxiv.org/abs/1703.06870) for 2D segmentation of a humanoid 
robotic hand. 

For the implementation of the network's architecture, we use the [matterport implementation](https://github.com/matterport/Mask_RCNN).
This library can be found in [my_mrcnn](https://github.com/alexalm4190/Mask_RCNN-Vizzy_Hand/tree/master/my_mrcnn).
The reason we include the library in our repository is due to some minor changes we did in their code.

In [hand](https://github.com/alexalm4190/Mask_RCNN-Vizzy_Hand/tree/master/hand) is where we have our main code to handle the 
datasets, extend some functions of [my_mrcnn](https://github.com/alexalm4190/Mask_RCNN-Vizzy_Hand/tree/master/my_mrcnn), configure 
hyperparameters and create the training/inference processes. 

In [utils](https://github.com/alexalm4190/Mask_RCNN-Vizzy_Hand/tree/master/utils) we also provide some utility functions, like
evaluation metrics, pre-processing functions, an annotation tool to generate groundtruth masks from real images, amongst other 
utilities.

To generate images for training and validation, we also provide a Unity framework to generate simulated images, available in 
[Unity_package](https://github.com/alexalm4190/Mask_RCNN-Vizzy_Hand/tree/master/Unity_package). 




