import cv2
import numpy as np
import math
import Tkinter as tk

class Tooth(object):
    def __init__(self, landmarks):
        self.landmarks = landmarks
        self._centroid = np.mean(self.landmarks, axis=0)
