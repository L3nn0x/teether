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
            return self.model.updatePosition(self.img, tooth)

    def __init__(self):
        self.resolutionLevels = []
        for i in range(0, MultiResolution.levelCount):
            self.resolutionLevels.append(MultiResolution.Resolution(MultiResolution.models[i]))

    def getLevel(self, level):
        return self.resolutionLevel[level]

    def addTrainingData(self, radiograph):
        img, (left, top, _, _) = radiograph.cropImage()
        teeth = radiograph.getTeeth()
        for tooth in teeth:
            tooth.translate(-(top, left))
        for i in range(0, MultiResolution.levelCount):
            resolution = self.resolutionLevels[i]
            if i > 0:
                img = cv2.pyrDown(img)
                for tooth in teeth:
                    tooth.scale(0.5)
            img = processImage(img, *MultiResolution.filter[i])
            resolution.model.addTrainingData(img, teeth)

    def setRadiograph(self, radiograph):
        img, (left, top, _, _) = radiograph.cropImage()
        self.crop = -(top, left)
        for i in range(0, MultiResolution.levelCount):
            if i > 0:
                img = cv2.pyrDown(img)
            img = processImage(img, *MultiResolution.filter[i])
            self.resolutionLevels[i].img = img
            self.resolutionLevels[i].radiograph = radiograph
