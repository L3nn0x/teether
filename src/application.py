from radiograph import Radiograph, processImage
import cv2

from activeshapemodel import ActiveShapeModel
from pca import PCA
from statisticmodel import create

class Application(object):
    def __init__(self):
        self.trainninRadiographs = [Radiograph("data/radiographs/01.tif", '1')]
        self.radiographs = []
        self.activeShapeModel = ActiveShapeModel(create(self.trainingRadiographs))

    def run(self):
        for radiograph in self.trainningRadiographs:
            self.activeShapeModel.train(radiograph)
        self.activeShapeModel.setup(self.radiographs[0])
        tooth = self.activeShapeModel.run()
