import cv2
import numpy as np
from operator import itemgetter

from radiograph import processImage

lowerJawSize = upperJawSize = sideSize = 160
maxAngle = 25
sideLinesThreshold = 100

def findJawLines(img):
    img = img[:, sideSize:img.shape[1] - sideSize]
    histogram = cv2.reduce(img, 1, cv2.cv.CV_REDUCE_SUM, dtype=cv2.CV_32S)
    threshold = 5000
    minIndex = np.argmin(histogram)
    minValue = histogram[minIndex]
    maxRangeIndex = minIndex
    for i in range(minIndex, minIndex + 200):
        if i > histogram.shape[0]:
            break
        if histogram[i] > minValue + threshold:
            break
        maxRangeIndex = i
    minRangeIndex = minIndex
    for i in reversed(range(minIndex - 200, minIndex)):
        if i < 0:
            break
        if histogram[i] > minValue + threshold:
            break
        minRangeIndex = i
    return minRangeIndex, maxRangeIndex

def cropJaw(img, y, upper):
    if not upper:
        return img[y:y + upperJawSize, sideSize:img.shape[1]-sideSize]
    return img[y-upperJawSize:y, sideSize:img.shape[1]-sideSize]

def toBinary(img):
    img = np.array(img, dtype=np.uint8)
    return cv2.threshold(img, 8, 255, cv2.THRESH_BINARY)[1]

def findHoughLines(img, threshold):
    return cv2.HoughLines(img, 1, 20*np.pi/180, threshold, 0, 0)#[i[0] for i in cv2.HoughLines(img, rho=1, theta=20*np.pi / 180, threshold=threshold)]

def filterLines(lines, shape, offset, maxGap):
    toConsider = []
    from math import radians
    for rho, theta in lines[0]:
        if (theta >= 0 and theta <= radians(maxAngle))\
           or (theta >= radians(180 - maxAngle) and theta <= radians(180)):
            toConsider.append((rho,theta))
    toConsider = sorted(toConsider, key=itemgetter(0))
    indices = []
    oldRho = 0
    oldTheta = 0
    oldId = 0
    for i, (rho, theta) in enumerate(toConsider):
        if rho < sideLinesThreshold or rho > shape[1] - sideLinesThreshold:
            continue
        elif rho < oldRho + maxGap and rho > oldRho - maxGap:
            if theta < oldTheta:
                oldId = i
                oldTheta = theta
        else:
            indices.append(oldId)
            oldRho = rho
            oldTheta = theta
            oldId = i
    if len(indices):
        indices.pop(0)
    indices.append(oldId)
    toConsider = np.array(toConsider)[indices]
    for i, (rho,theta) in enumerate(toConsider):
        toConsider[i] = (rho - offset, theta)
    middle = shape[1] / 2
    minId = 0
    minDist = float('inf')
    for i, (rho,theta) in enumerate(toConsider):
        if abs(rho - middle) < minDist:
            minDist = abs(rho - middle)
            minId = i
    new = toConsider[minId - 1:minId + 2]
    if new.shape[0] != 3:
        rho, theta = toConsider[minId]
        lines = [(rho+maxGap,0),(rho,theta),(rho-maxGap,0)]
    else:
        lines = new
    return np.array(sorted(lines, key=itemgetter(0)))

def printRhoTheta(img, rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0+1000*(-b))
    y1 = int(y0+1000*a)
    x2 = int(x0-1000*(-b))
    y2 = int(y0-1000*a)
    cv2.line(img, (x1,y1),(x2,y2),(255,0,0))

def findInitialTeeth(radiograph):
    img = radiograph.getImage()
    upperJawLine, lowerJawLine = findJawLines(img)
    upperJaw = cropJaw(img, upperJawLine, True)
    lowerJaw = cropJaw(img, lowerJawLine, False)

    upperJaw = processImage(upperJaw, 5, 17, 6)
    lowerJaw = processImage(lowerJaw, 5, 17, 6)

    upperJaw = toBinary(upperJaw)
    lowerJaw = toBinary(lowerJaw)

    upperLines = findHoughLines(upperJaw, 15)
    lowerLines = findHoughLines(lowerJaw, 15)

    upperLines = filterLines(upperLines, upperJaw.shape, 6, 90)
    lowerLines = filterLines(lowerLines, lowerJaw.shape, 6, 90)

    rho,theta = zip(*upperLines)
    pos = [np.array((rho[0]-35+sideSize,50+upperJawLine-upperJawSize))]
    for i in range(1,3):
        pos.append(np.array((rho[i-1]+(rho[i]-rho[i-1])/2+sideSize,80+upperJawLine-upperJawSize)))
    pos.append(np.array((rho[2]+25+sideSize,80+upperJawLine-upperJawSize)))
    rho,theta = zip(*lowerLines)
    pos.append(np.array((rho[0]-40+sideSize,50+lowerJawLine)))
    for i in range(1,3):
        pos.append(np.array((rho[i-1]+(rho[i]-rho[i-1])/2+sideSize,90+lowerJawLine)))
    pos.append(np.array((rho[2]+sideSize,90+lowerJawLine)))

    return zip(pos,
            (48, 55, 55, 48, 40, 38, 38, 40),
            (0.05, 0.2, 0.2, 0.3, 0, -0.05, -0.1, -0.15))
