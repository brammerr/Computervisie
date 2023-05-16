import os
import cv2
import numpy as np

def ResizeImage(img):
    return cv2.resize(img, [int(img.shape[1] / img.shape[0] * 400), 400], cv2.INTER_AREA)
