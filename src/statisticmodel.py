from pca import PCA
from tooth import Tooth

from copy import deepcopy
import numpy as np

def create(trainingRadiographs, nbTooth, componentsNb=0):
    print("GPA & PCA analysis for tooth", nbTooth)
    teeth = []
    for radio in trainingRadiographs:
        teeth.append(deepcopy(radio.teeth[nbTooth]))
        teeth[-1].translateToOrigin()
        teeth[-1].normalize()
    mean = deepcopy(teeth[0])
    error = float('inf')
    while error > 0.000001:
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
