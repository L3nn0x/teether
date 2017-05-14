import numpy as np

def findPositions(center, normal, count):
    pos = []
    neg = []
    center = tuple(center.astype(np.int32))
    scale = 0
    last = center
    while len(pos) < count:
        scale += 0.5
        sample = tupple(np.floor(center + normal * scale).astype(np.int32))
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

def sample(tooth, img, k, normalize=False):
    result = np.empty((tooth.landmarks.shape[0], 2 * k, k + 1))
    for i, center in enumerate(tooth.landmarks):
        normal = tooth.normals[i]
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
    return result

class IntensityModel(object):
    def __init__(self, k=2, m=12, normalize=True):
        self.k = k
        self.m = m
        self.normalize = normalize
        self.samples = []

    def addTrainingData(self, teeth, img):
        for i, tooth in enumerate(teeth):
            pass # TODO
