from radiograph import processImage

from intensitymodel import IntensityModel

class MultiResolution(object):
    levelCount = 2 # max number of levels in the gaussian pyramid
    models = ((5, 14), (5, 14), (2, 5))
    filter = ((5, 17, 6), (3, 15, 6), (0, 7, 6))

    class Resolution(object):
        def __init__(self, model):
            self.model = IntensityModel(*model)

        def updateLandmarks(self, tooth):
            return self.model.updatePositions(tooth, self.img)

    def __init__(self):
        self.resolutionLevels = []
        for i in range(0, MultiResolution.levelCount):
            self.resolutionLevel.append(MultiResolution.Resolution(MultiResolution.models[i]))

    def addTrainingData(self, radiograph):
        img, (translation, _, _, _) = radiograph.cropImage()
        teeth = radiograph.getTeeth()
        for tooth in teeth:
            tooth.translate(-translation)
        for i in range(0, MultiResolution.levelCount):
            resolution = self.resolutionLevels[i]
            if i > 0:
                img = cv2.pyrDown(img)
                for tooth in teeth:
                    tooth.scale(0.5)
            img = processImage(img, *MultiResolution.filter[i])
            resolution.model.addTrainingData(img, teeth)

    def train(self):
        for resolution in self.resolutionLevels:
            resolution.model.train()
