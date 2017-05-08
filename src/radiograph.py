import cv2
import numpy as np

class Radiograph(object):
    def __init__(self, filename, hasLandmark=False):
        self._teeth = list()
        self.path = filename
