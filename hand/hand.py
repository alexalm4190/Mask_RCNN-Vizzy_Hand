#python libraries
import os
import argparse
import sys

#my libraries
from datasets import Datasets
from mrcnn_extensions import HandConfig, HandDataset
from mrcnn_model import Model

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

#mrcnn libraries
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils

device = "/gpu:0"

# Initialize the parser
parser = argparse.ArgumentParser()
# Add arguments to the parser
parser.add_argument('-m', "--mode", 
					help='choose which mode to run Mask RCNN, train or test',
					required=True)
parser.add_argument('-d', "--modelDir", 
					help='choose the directory in which to save the model logs, during training',
					required=True)						
parser.add_argument('-i', "--imageDir", 
					help='choose the directory, where the image Datasets are stored',
					required=True)
parser.add_argument('-w', "--weightsPath", 
					help='Path to the pre-trained weights',
					required=True)				
parser.add_argument('-t', "--testDataset",
					help='use a test dataset')
parser.add_argument('-p', "--modelPath", 
					help='Path of the model to load when testing',
					required=False)
# Parse the arguments
arguments = parser.parse_args()

if (arguments.modelPath == None) and (arguments.mode != "train"):
	print("Missing arg: You have to specify which model you want to load, to test the network.")
	sys.exit()

config = HandConfig()
config.display()

myDatasets = Datasets(arguments.imageDir, arguments.testDataset, trainSplit=0.8)
imagePaths, masksPath, testImagePaths, testMasksPath = myDatasets.split_indexes()

dataset_train = HandDataset(imagePaths, masksPath, arguments.testDataset, testImagePaths, testMasksPath)
dataset_val = HandDataset(imagePaths, masksPath, arguments.testDataset, testImagePaths, testMasksPath)
if arguments.testDataset:
	dataset_test = HandDataset(imagePaths, masksPath, arguments.testDataset, testImagePaths, testMasksPath)
else:
	dataset_test = None	

dataset_train, dataset_val, dataset_test = myDatasets.prepare_datasets(dataset_train, dataset_val, config.IMAGE_SHAPE, dataset_test)

class InferenceConfig(HandConfig):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

model = Model(arguments.modelDir, arguments.weightsPath, device, 
			  dataset_train, dataset_val, dataset_test)

if arguments.mode == "train":
	model.train_model(config)
else:
	inference_config = InferenceConfig()
	#model_path = model.find_last()	
	model.test_model(inference_config, arguments.modelPath)
	

