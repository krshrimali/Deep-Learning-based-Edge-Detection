"""
Performs Deep Learning based Edge Detection using HED (Holistically Nested Edge Detection)

HED uses Trimmed VGG-like CNN (for image to prediction)

Author: krshrimali
Motivation: https://cv-tricks.com/opencv-dnn/edge-detection-hed/ (by Ankit Sachan)
"""

import cv2 as cv
import numpy as np
from utils.preprocessing import CannyP
from utils.preprocessing import CropLayer

import sys

if __name__ == "__main__":
    # get image path
    if(len(sys.argv) > 1):
        src_path = sys.argv[1]
    else:
        src_path = "testdata/source-image.jpg"

    # read image
    img = cv.imread(src_path, 1) 
    if(img is None):
        print("Image not read properly")
        sys.exit(0)

    # initialize preprocessing object
    obj = CannyP(img)
    
    width = 500
    height = 500
    
    # remove noise
    img = obj.noise_removal(filterSize=(5, 5))
    prototxt = "../Edge-Detection-Deep-Learning/deploy.prototxt"
    caffemodel = "../Edge-Detection-Deep-Learning/hed_pretrained_bsds.caffemodel"

    cv.dnn_registerLayer('Crop', CropLayer)
    net = cv.dnn.readNet(prototxt, caffemodel)

    inp = cv.dnn.blobFromImage(img, scalefactor=1.0, size=(width, height), \
            mean=(104.00698793, 116.66876762, 122.67891434), \
            swapRB=False, crop=False)

    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = cv.resize(out, (img.shape[1], img.shape[0]))
    out = 255 * out
    out = out.astype(np.uint8)
    # out = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    # con = np.concatenate((img, out), axis=1)
    cv.imshow("HED", out)
    cv.imshow("original", img)

    # visualize
    cv.imshow("Original Image", img)
    cv.imshow("Image w/o Noise", img)

    k = cv.waitKey(0) & 0xFF
    if(k == 27):
        cv.destroyAllWindows()
