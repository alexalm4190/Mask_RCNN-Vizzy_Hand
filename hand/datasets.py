#python libraries
import os
import random
import glob
from imutils import paths

#my libraries
from mrcnn_extensions import HandConfig, HandDataset

class Datasets():

    def __init__(self, image_dir, testDataset=False, trainSplit=0.8):
        self.image_dir = image_dir
        self.testDataset = testDataset
        self.trainSplit = trainSplit

    def split_indexes(self):
        imagesPath = os.path.sep.join([self.image_dir, "train_images/images"])
        masksPath = os.path.sep.join([self.image_dir, "train_images/mask"])

        testImagePaths = None
        testMasksPath = None
        if self.testDataset:
            testImagesPath = os.path.sep.join([self.image_dir, "test_images/images"])
            testMasksPath = os.path.sep.join([self.image_dir, "test_images/mask"])
            testImagePaths = sorted(list(paths.list_images(testImagesPath)))
            self.testIdxs = list(range(0, len(testImagePaths)))

        imagePaths = [f for f in glob.glob(imagesPath + "/*.png")]
        print(imagePaths)
        negImagePaths = [f for f in glob.glob(imagesPath + "/*.jpg")]
        print(negImagePaths)
        self.trainIdxs = list(range(0, len(imagePaths)))
        idxs = list(range(0, len(negImagePaths)))
        random.seed(None)
        random.shuffle(idxs)
        i = int(len(self.trainIdxs) * (1-self.trainSplit))
        self.valIdxs = idxs[:i]
        #imagePaths = sorted(list(paths.list_images(imagesPath)))
        #idxs = list(range(0, len(imagePaths)))
        #random.seed(None)
        #random.shuffle(idxs)
        #i = int(len(idxs) * self.trainSplit)
        #self.trainIdxs = idxs[:i]
        #self.valIdxs = idxs[i:]

        return imagePaths, negImagePaths, masksPath, testImagePaths, testMasksPath

    def prepare_datasets(self, dataset_train, dataset_val, imageShape, dataset_test=None):
        #Training dataset
        dataset_train.load_hands(self.trainIdxs, imageShape[0], imageShape[1])
        dataset_train.prepare()

        # Validation dataset
        dataset_val.load_hands(self.valIdxs, imageShape[0], imageShape[1])
        dataset_val.prepare()

        # Test dataset
        if dataset_test != None:
            dataset_test.load_hands_test(self.testIdxs, imageShape[0], imageShape[1])
            dataset_test.prepare()    

        return dataset_train, dataset_val, dataset_test