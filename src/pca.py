from landmarks import pca
import numpy as np

class PCA(object):
    def __init__(self, X, numComponents=0):
        self.eigenValues, self.eigenVectors, self.mean = pca(X, numComponents)

    def project(self, X):
        return np.dot(X - self.mean, self.eigenVectors)

    def getMaxDeviation(self):
        return 2 * np.sqrt(self.eigenValues)
    
    def reconstruct(self, b):
        return np.dot(b, self.eigenVectors.T) + self.mean

    def limit(self, value):
        value = min(1, max(0, value))
        variance = self.eigenValues / sum(self.eigenValues)
        cumulativeVariance = np.cumsum(variance)
        i = 0
        while i < len(self.eigenValues) - 1 and cumulativeVariance[i] < value:
            i += 1
        self.eigenValues = self.eigenValues[:i + 1]
        self.eigenVectors = self.eigenVectors[:i + 1]
