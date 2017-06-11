from radiograph import Radiograph, processImage
import cv2

from activeshapemodel import ActiveShapeModel
from pca import PCA
from statisticmodel import create

from initialpose import findInitialTeeth

class Application(object):
    def __init__(self):
        print("loading {} training radiographs".format(14-1))
        self.trainingRadiographs = list(Radiograph("data/radiographs/{:02}.tif".format(i), str(i), True) for i in range(1, 14))
        print("done")
        print("loading {} test radiographs".format(30-15))
        self.radiographs = list(Radiograph("data/radiographs/extra/{}.tif".format(i), str(i)) for i in range(15, 30))
        print("done")
        pca = [create(self.trainingRadiographs, i) for i in range(8)]
        print("done")
        for p in pca:
            p.limit(0.5)
        self.activeShapeModel = ActiveShapeModel(pca)

    def run(self):
        print("training model")
        for radiograph in self.trainingRadiographs:
            self.activeShapeModel.train(radiograph)
        print("done")

        radiograph = self.radiographs[0]
        print("finding initial poses")
        poses = findInitialTeeth(radiograph)
        print("done")
        for i, pose in enumerate(poses):
            print("computing tooth",i)
            self.activeShapeModel.setup(radiograph, i, *pose)
            radiograph.teeth.append(self.activeShapeModel.run())
            print("done")
        print("writing file")
        radiograph.writeImg("radiograph0.png", poses)
        print("done")
