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
        testIdxs = None
        if self.testDataset:
            testImagesPath = os.path.sep.join([self.image_dir, "test_images/images"])
            testMasksPath = os.path.sep.join([self.image_dir, "test_images/mask"])
            testImagePaths = sorted(list(paths.list_images(testImagesPath)))
            testIdxs = list(range(0, len(testImagePaths)))

        imagePaths = [f for f in glob.glob(imagesPath + "/*.png")]
        negImagePaths = [f for f in glob.glob(imagesPath + "/*.jpg")]
        idxs = list(range(0, len(negImagePaths)))
        random.seed(None)
        random.shuffle(idxs)
        for i in range(0, 300):
            imagePaths.append(negImagePaths[idxs[i]])
        idxs = list(range(0, len(imagePaths)))
        random.shuffle(idxs)
        i = int(len(idxs) * self.trainSplit)
        trainIdxs = idxs[:i]
        valIdxs = idxs[i:]
        #imagePaths = sorted(list(paths.list_images(imagesPath)))
        #idxs = list(range(0, len(imagePaths)))
        #random.seed(None)
        #random.shuffle(idxs)
        #i = int(len(idxs) * self.trainSplit)
        #self.trainIdxs = idxs[:i]
        #self.valIdxs = idxs[i:]

        return imagePaths, masksPath, testImagePaths, testMasksPath, trainIdxs, valIdxs, testIdxs

    def prepare_dataset(self, dataset, idxs, imageShape):

        dataset.load_hands(idxs, imageShape[0], imageShape[1])
        dataset.prepare()

        return dataset