from radiograph import Radiograph, processImage
import cv2

from activeshapemodel import ActiveShapeModel
from pca import PCA
from statisticmodel import create

from initialpose import findInitialTeeth

class Application(object):
    def __init__(self):
        self.trainingRadiographs = list(Radiograph("data/radiographs/{:02}.tif".format(i), str(i), True) for i in range(1, 14))
        self.radiographs = list(Radiograph("data/radiographs/extra/{}.tif".format(i), str(i)) for i in range(15, 30))
        pca = [create(self.trainingRadiographs, i) for i in range(8)]
        for p in pca:
            p.limit(0.5)
        self.activeShapeModel = ActiveShapeModel(pca)

    def run(self):
        for radiograph in self.trainingRadiographs:
            self.activeShapeModel.train(radiograph)
        radiograph = self.radiographs[0]
        poses = findInitialTeeth(radiograph)
        for i, pose in enumerate(poses):
            self.activeShapeModel.setup(radiograph, i, *pose)
            radiograph.teeth.append(self.activeShapeModel.run())
        radiograph.writeImg("radiograph0.png")
