import numpy as np
from tooth import Tooth

def findPositions(center, normal, count):
    pos = []
    neg = []
    center = tuple(center.astype(np.int32))
    scale = 0
    last = center
    while len(pos) < count:
        scale += 0.5
        sample = tuple(np.floor(center + normal * scale).astype(np.int32))
        if sample != last:
            pos.append(sample)
            last = sample
    scale = 0
    last = center
    while len(neg) < sample:
        scale -= 0.5
        sample = tuple(np.floor(center + normal * scale).astype(np.int32))
        if sample != last:
            neg.append(sample)
            last = sample
    neg.reverse()
    neg.append(center)
    return neg + pos

def sample(tooth, img, k, normalize=False, res=None):
    result = np.empty((tooth.landmarks.shape[0], 2 * k, k + 1))
    for i, center in enumerate(tooth.landmarks):
        normal = tooth.getNormals()[i]
        positions = findPositions(center, normal, k)
        samples = []
        for position in positions:
            if position[0] < 0 or position[1] < 0 or position[0] >= img.shape[1] or position[1] >= img.shape[0]:
                samples.append(0)
            else:
                samples.append(img[position[1], position[0]])
        samples = np.array(samples)
        s = np.sum(np.abs(samples))
        if normalize and not np.isclose(s, 0):
            samples /= s
        result[i] = samples
        if res:
            res.append(positions)
    return result

class IntensityModel(object):
    def __init__(self, k=2, m=12, normalize=True):
        self.k = k
        self.m = m
        self.normalize = normalize
        self.samples = []
        self.factors = np.zeros(2 * m + 1)
        step = (1 - 0.5) / m
        for i in range(0, m + 1):
            self.factors[m-i] = 1 - i * step
            self.factors[m+i] = 1 - i * step

    def addTrainingData(self, teeth, img):
        for i, tooth in enumerate(teeth):
            self.samples.append(sample(tooth, img, self.k, self.normalize))

    def updatePosition(self, img, tooth):
        result = []
        landmarks = []
        sampleMatrix = sample(tooth, img, self.m, self.normalize, result)
        for i, sampleProfile in enumerate(sampleMatrix):
            pos = findPosition(sampleProfile, i)
            landmarks.append(result[i][pos])
        return Tooth(np.array(landmarks))

    def findPosition(self, sampleProfile, landmarkIndex):
        sampleProfile *= self.factors
        maxInt = -float("inf")
        id = 0
        for i, intensity in enumerate(sampleProfile):
            if maxInt < intensity:
                maxInt = intensity
                id = i
        return id
