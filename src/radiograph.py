import numpy as np
import cv2
import landmarks as lm
from tooth import Tooth
from copy import deepcopy

def scharr(image):
    """Applies scharr gradient to image"""
    gradX = cv2.Scharr(image, cv2.CV_64F, 1, 0) / 16
    gradY = cv2.Scharr(image, cv2.CV_64F, 0, 1) / 16
    return np.sqrt(gradX ** 2, gradY ** 2)

def processImage(image, medianKernel=5, bilateralKernel=17, bilateralColor=9):
    """filters image by using median & bilateral filters followed by scharr"""
    image = cv2.medianBlur(image, medianKernel)
    image = cv2.bilateralFilter(image, bilateralKernel, bilateralColor, 200)
    return scharr(image)

class Radiograph(object):
    def __init__(self, filename, radioID, hasLandmark=False):
        self.teeth = []
        self.filename = filename
        self.image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
        if hasLandmark:
            directory = "data/landmarks/original"
            self.teeth.append(Tooth(lm.getLandmarks2(directory, radioID)))

    def getImg(self):
        return self.image.copy()

    def cropImage(self):
        _, w = self.image.shape
        left, top, right, bottom = (w/2 - 400, 500, w/2 + 400, 1400)
        return self.image[top:bottom, left:right].copy(), (left, top, right, bottom)

    def getTeeth(self):
        return deepcopy(self.teeth)

    def writeImg(self, name):
        for tooth in teeth:
            for landmark in tooth.landmarks:
                points = lm.landmarkAsMatrix(landmark)
                for i in range(len(points) - 1):
                    cv2.line(img, (int(points[i, 0]), int(points[i, 1])), (int(points[i+1, 0]), int(points[i+1, 1])), (255, 255, 0))
                cv2.line(img, (int(points[0, 0]), int(points[0, 1])), (int(points[len(points) - 1, 0]), int(points[len(points) - 1, 1])), (255, 255, 0))
        
            points = lm.landmarkAsMatrix(lm.getMeanShape(landmarks))
            for i in range(len(points) - 1):
                cv2.line(img, (int(points[i, 0]), int(points[i, 1])), (int(points[i+1, 0]), int(points[i+1, 1])), (255, 0, 0))
            cv2.line(img, (int(points[0, 0]), int(points[0, 1])), (int(points[len(points) - 1, 0]), int(points[len(points) - 1, 1])), (255, 0, 0))
        cv2.imgwrite(name, self.image)
