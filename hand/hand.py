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
from my_mrcnn import utils

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
imagePaths, masksPath, testImagePaths, testMasksPath, trainIdxs, valIdxs, testIdxs = myDatasets.split_indexes()

if arguments.testDataset:
	dataset_test = HandDataset(testImagePaths, testMasksPath)
	dataset_test = myDatasets.prepare_dataset(dataset_test, testIdxs, config.IMAGE_SHAPE)
else:	
	dataset_train = HandDataset(imagePaths, masksPath)
	dataset_val = HandDataset(imagePaths, masksPath)
	dataset_train = myDatasets.prepare_dataset(dataset_train, trainIdxs, config.IMAGE_SHAPE)
	dataset_val = myDatasets.prepare_dataset(dataset_val, valIdxs, config.IMAGE_SHAPE)

	display = "Number of train images: " + str(len(dataset_train.image_info))
	print(display)
	display = "Number of val images: " + str(len(dataset_val.image_info))
	print(display)

class InferenceConfig(HandConfig):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

model = Model(arguments.modelDir, device)

if arguments.mode == "train":
	model.train_model(config, arguments.weightsPath, dataset_train, dataset_val)
else:
	inference_config = InferenceConfig()
	#model_path = model.find_last()	
	model.test_model(inference_config, arguments.modelPath, dataset_test)
	

