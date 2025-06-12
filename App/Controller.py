from Data.FilterSegmentation import FilterSegmentation
import numpy as np
import torch
import os
from DLPatch.DLModel import DLModel
from torchvision import transforms as T
from PIL import Image
import cv2

from Data.Preprocessing import ImagePreprocessing

class Controller:
    def run_filter(self, image):
        return FilterSegmentation.run(image)

    def run_ml(self, image):
        return self.dummy(image)

    def run_dl(self, image):
        return self.dummy(image)
    
    def dummy(self, image):
        image = ImagePreprocessing.resize_and_normalize(image)
        image = (image > 0).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray
    