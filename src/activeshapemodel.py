from multiresolution import MultiResolution
from tooth import Tooth

from copy import deepcopy
import cv2
import numpy as np

class ActiveShapeModel(object):
    maxStepsPerLevel = 100

    def __init__(self, pca):
        self.pca = pca
        self.multiResolution = MultiResolution()
        self.currentLevel = 0
    
    def train(self, radiograph):
        self.multiResolution.addTrainingData(img)

    def setup(self, radiograph, translation=(0, 0), scale=1, rotation=0):
        self.multiResolution.setRadiograph(radiograph)
        mean = self.pca.mean
        self.meanTooth = Tooth(mean.reshape(int(mean.size / 2), 2))
        self.currentTooth = deepcopy(self.meanTooth)
        self.currentTooth.transform(translation, scale, rotation)
        self.currentParams = np.zeros(self.pca.eigenValues.shape)

    def step(self):
        resolutionLevel = self.multiResolution.getLevel(self.currentLevel)
        tooth = resolutionLevel.updateLandmarks(self.currentTooth)
        translation, scale, rotation = tooth.align(self.meanTooth)
        b = self.pca.project(tooth.landmarks.flatten())
        maxDeviation = self.pca.getMaxDeviation()
        for i in range(0, b.shape[0]):
            b[i] = min(max(b[i], -maxDeviation[i]))
        scale = min(max(scale, 5), 80 / (2 ** self.currentLevel))
        shape = self.pca.reconstruct(b)
        tooth = Tooth(shape.reshape(int(shape.size / 2), 2))
        tooth.transform(translation, scale, rotation)
        self.currentParams = b
        self.currentTooth = tooth

    def run(self):
        self.currentLevel = 0
        level = MultiResolutionFramework.levelCounts - 1
        while level >= 0:
            diff = level - self.currentLevel
            self.currentLevel = level
            if diff < 0:
                for i in range(0, abs(diff)):
                    self.currentTooth.upSample()
            else:
                for i in range(0, abs(diff)):
                    self.currentTooth.downSample()
            steps = ActiveShapeModel.maxStepsPerLevel
            while steps > 0:
                previousTooth = self.currentTooth
                self.step()
                diff = self.currentTooth.diff(previousTooth)
                if diff < 1:
                    self.currentTooth = previousTooth
                    break
                steps -= 1
            level -= 1
        res = deepcopy(self.currentTooth)
        res.translate(-self.multiResolution.crop)
        return res
