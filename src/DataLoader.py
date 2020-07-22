from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2

from SamplePreprocessor import preprocess

class Sample:
    "Sample from the dataset"
    def__init__(self, gtText, filePath):
    self.gtText = gtText
    self.filePath = filePath


class Batch:
    "batch containing images and ground truth texts"
    def __init__ (self, gtText, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts

class DataLoader:
    
    def __init__(self, filePath, batchSize, imgSize, maxTextLen):

        assert filePath[-1] == '/'

        self.dataAugmentation = False
        self.currIdx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []

        f=open(filePath+'words.txt')
        chars = set()
        