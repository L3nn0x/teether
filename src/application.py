from radiograph import Radiograph, processImage
import cv2

class Application(object):
    def __init__(self):
        self.radiographs = [Radiograph("data/radiographs/01.tif", '1')]

    def run(self):
        image = self.radiographs[0].cropImage()
        image = processImage(image)
        cv2.imwrite('img.jpg', image)
