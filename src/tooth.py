import cv2
import numpy as np
import math

from landmarks import rotationMatrix

class Tooth(object):
    def __init__(self, landmarks):
        self.landmarks = landmarks
        self._centroid = self.computeCentroid()
        self._normals = None

    def getCentroid(self):
        if self._centroid is None:
            self._centroid = self.computeCentroid()
        return self._centroid

    def getNormals(self):
        if self._normals is None:
            self.computeNormals()
        return self._normals

    def computeCentroid(self):
        return np.mean(self.landmarks, axis=0)

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
        nLeft /= np.linalg.norm(nLeft, axis=1).reshape(40,1)
        nRight /= np.linalg.norm(nRight, axis=1).reshape(40,1)
        self._normals = (nLeft + nRight)
        self._normals /= np.linalg.norm(self._normals, axis=1).reshape(40, 1)

    def align(self, other):
        #procrustes analysis
        translation = self.translateToOrigin()
        scale = self.normalize()
        x = self.landmarks[:, 0]
        y = self.landmarks[:, 1]
        u = other.landmarks[:, 0]
        v = other.landmarks[:, 1]
        tSum = np.sum(u * y - v * x)
        bSum = np.sum(u * x + v * y)
        angle = math.atan(tSum / bSum)
        self.rotate(angle)
        return translation, scale, -angle

    def transform(self, translation, scale, rotation):
        self.rotate(rotation)
        self.scale(scale)
        self.translate(translation)

    def upSample(self):
        self.scale(2)

    def downSample(self):
        self.scale(0.5)

    def diff(self, other):
        return np.sum((self.landmarks - other.landmarks) ** 2)

    def translate(self, translation):
        translation = [translation]*len(self.landmarks)
        self.landmarks += translation
        self._centroid = None

    def scale(self, scale):
        translation = self.translateToOrigin()
        self.landmarks *= scale
        self.translate(translation)
        self._normals = None
        self._centroid = None

    def rotate(self, theta):
        rotMatrix = rotationMatrix(theta)
        self.landmarks = self.landmarks.dot(rotMatrix)
        self._normals = None
        self._centroid = None

    def normalize(self):
        scale = self.landmarks - self.getCentroid()
        scale = np.sum(scale ** 2)
        scale = np.sqrt(scale / self.landmarks.size)
        self.scale(1 / scale)
        return scale

    def translateToOrigin(self):
        translation = self.getCentroid()
        self.translate(-translation)
        return translation
