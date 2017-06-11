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
        self.multiResolution.addTrainingData(radiograph)

    def setup(self, radiograph, nbTooth, translation=(0, 0), scale=1, rotation=0):
        self.nbTooth = nbTooth
        self.multiResolution.setRadiograph(radiograph)
        mean = self.pca[nbTooth].mean
        self.meanTooth = Tooth(mean.reshape(int(mean.size / 2), 2))
        self.currentTooth = deepcopy(self.meanTooth)
        self.currentTooth.transform(translation, scale, rotation)
        self.currentParams = np.zeros(self.pca[nbTooth].eigenValues.shape)

    def step(self, write):
        resolutionLevel = self.multiResolution.getLevel(self.currentLevel)
        tooth = resolutionLevel.updateLandmarks(self.currentTooth)
        translation, scale, rotation = tooth.align(self.meanTooth)
        b = self.pca[self.nbTooth].project(tooth.landmarks.flatten())
        maxDeviation = self.pca[self.nbTooth].getMaxDeviation()
        for i in range(0, b.shape[0]):
            b[i] = min(max(b[i], -maxDeviation[i]), maxDeviation[i])
        scale = min(max(scale, 5), 80 / (2 ** self.currentLevel))
        shape = self.pca[self.nbTooth].reconstruct(b)
        tooth = Tooth(shape.reshape(int(shape.size / 2), 2))
        tooth.transform(translation, scale, rotation)
        self.currentParams = b
        self.currentTooth = tooth

    def run(self, write):
        self.currentLevel = 0
        level = MultiResolution.levelCount - 1
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
                self.step(write)
                diff = self.currentTooth.diff(previousTooth)
                if diff < 1:
                    self.currentTooth = previousTooth
                    break
                steps -= 1
            level -= 1
        res = deepcopy(self.currentTooth)
        #res.translate((-self.multiResolution.top, -self.multiResolution.left))
        return res
