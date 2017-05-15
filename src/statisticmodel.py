from pca import PCA
from tooth import Tooth

from copy import deepcopy
import numpy as np

def create(trainingRadiographs, componentsNb=0):
    teeth = []
    for radio in trainingRadiographs:
        teeth.append(deepcopy(i) for i in radio.teeth)
    mean = deepcopy(teeth[0])
    mean.translateToOrigin()
    mean.normalize()
    error = float('inf')
    while error > 0.05:
        meanAcc = np.zeros(mean.landmarks.shape)
        for tooth in teeth:
            tooth.align(mean)
            meanAcc += tooth.landmarks
        nMean = Tooth(meanAcc / len(teeth))
        nMean.align(mean)
        error = nMean.diff(mean)
        mean = nMean
    for tooth in teeth:
        tooth.align(mean)
    data = np.zeros((len(teeth), teeth[0].landmarks.size))
    for i, tooth in enumerate(teeth):
        data[i, :] = tooth.landmarks.flatten()
    return PCA(data, componentsNb)
