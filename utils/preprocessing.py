"""
Utilities to preprocess image used for Canny Edge Detection
and Deep Learning based Edge Detection

Author: krshrimali
Motivation: https://cv-tricks.com/opencv-dnn/edge-detection-hed/ (by Ankit Sachan)
"""

import cv2 as cv
import numpy as np

class CannyP:
    """
    Preprocessing methods for Canny Edge Detection
    """
    def __init__(self, img):
        self.img = img
    def noise_removal(self, filterSize=(5, 5)):
        """
        If there are sudden changes in intensity, it canny detects as an edge.
        Therefore, requirement of removal of noise.

        Noise Removal using filter size of (5, 5) [kernel size]
        ====================
        Parameters
        - filterSize: (tuple) default set to (5, 5)
        ====================
        Returns
        - image with noise removed
        """
        # apply Gaussian Kernel of size filterSize to the image
        blur = cv.GaussianBlur(self.img, filterSize, 0)        
        # return blurred image
        return blur

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2])//2
        self.xstart = (inputShape[3] - targetShape[3])//2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]

