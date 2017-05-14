import cv2
import numpy as np
import math

class Tooth(object):
    def __init__(self, landmarks):
        self.landmarks = landmarks
        self.centroid = np.mean(self.landmarks, axis=0)
        self.normals = None

    def computeNormals(self):
        nbPoints = self.landmarks.shape[0]
        nLeft = np.empty(self.landmarks.shape)
        nRight = np.empty(self.landmarks.shape)
        def normal(pt1, pt2):
            vec = (pt1 - pt2)
            return vec[1], -vec[0]
        for i in range(0, len(self.landmarks)):
            nLeft[i] = normal(self.landmarks[i - 1], self.landmarks[i])
            nRight[i] = normal(self.landmarks[i], self.landmarks[(i + 1) % nbPoints])
        nLeft /= np.linalg.norm(nLeft, axis=1).reshape(40, 1)
        nRight /= np.linalg.norm(nRight, axis=1).reshape(40, 1)
        self.normals = (nLeft + nRight)
        self.normals /= np.linalg.norm(self.normals, axis=1).reshape(40, 1)
