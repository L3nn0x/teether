import cv2
import os
import numpy as np
import fnmatch
import csv
import math
import matplotlib.pyplot as plt

def pca(landmarks, nb_components=0):
    '''
    Do a PCA analysis on landmarks
    @param landmarks:                np.array containing the samples
                             shape = (nb samples, nb dimensions of each sample)
    @param nb_components:    the nb components we're interested in
    @return: return the nb_components largest eigenvalues and eigenvectors of the covariance matrix and return the average sample
    '''
    #[n,d1,d2] = landmarks.shape
    #if (nb_components <= 0) or (nb_components>d1*d2):
    #    nb_components = d1*d2

    #landmarks2 = np.zeros((n, d1*d2))
    #for i in range(len(landmarks)):
    #    landmarks2[i] = landmarkAsVector(landmarks[i])

    landmarks2 = landmarks
    [n, d] = landmarks2.shape
    if (nb_components <= 0) or (nb_components > d):
        nb_components = d

    mu = getMeanShape(landmarks2)
    landmarks2 -= mu

    cov_mat = np.cov(landmarks2.transpose())
    eig_val, eig_vec = np.linalg.eig(cov_mat)

    eig = [list(x) for x in zip(eig_val, eig_vec.transpose())]
    eig.sort(key=lambda tup: tup[0], reverse=True)
    eig = eig[:nb_components]
    eig_val, eig_vec = zip(*eig)

    eig_val = np.array(eig_val)
    eig_vec = np.array(eig_vec)
    eig_vec = eig_vec.transpose()

    eig_vec = landmarks2.dot(eig_vec)

    for i in eig_vec.transpose():
        i/=np.linalg.norm(i)

    return  eig_val, eig_vec, mu

def getLandmarks(directory, tooth):
    """
        This function gets the landmarks for the given tooth.
        params:
            directory : folder with landmarks files
            tooth : string identifier of tooth
    """

    landmarksArray = []
    for filename in fnmatch.filter(os.listdir(directory),'*-'+tooth+'.txt'):
        with open(directory+"/"+filename) as landmarkFile:
            landmarkVector = np.array(landmarkFile.readlines(), dtype=float)
            landmarksArray.append(landmarkVector)

    return np.array(landmarksArray)

def getLandmarks2(directory, radiograph):
    """
        This function gets the landmarks for all 8 teeth for the given radiograph.
        params:
            directory : folder with landmarks files
            radiograph : string identifier of radiograph
    """

    landmarksArray = []
    for filename in fnmatch.filter(os.listdir(directory),'landmarks'+radiograph+'-*'):
        with open(directory+"/"+filename) as landmarkFile:
            landmarkVector = np.array(landmarkFile.readlines(), dtype=float)
            landmarksArray.append(landmarkVector)

    return np.array(landmarksArray)

def landmarkAsMatrix(landmarkVector):
    """ This function returns the landmark vector as a matrix."""

    return landmarkVector.reshape((landmarkVector.shape[0] / 2, 2))

def landmarkAsVector(landmarkMatrix):
    """ This function returns the landmark matrix as a vector."""

    return np.hstack(landmarkMatrix)

def translateToOrigin(points):
    """ This function translates the points so their center of gravity is at the origin."""

    return points - sum(points) / points.shape[0]

def scaleTo1(points):
    """ This function scales the points that are assumed to be already centered around the origin."""

    scale = sum(pow(sum(pow(points, 2.0) / float(points.shape[0])), 0.5))
    scale3 = np.linalg.norm(points)

    return points / scale3

def getScalingFactor(points, template):
    """ This function returns the sclaing factor that best matches the points against the template. """

    x1 = landmarkAsVector(points)
    x2 = landmarkAsVector(template)
    a = np.dot(x1, x2) / np.linalg.norm(x1)**2
    b = (np.dot(x1[:len(x1)/2], x2[len(x2)/2:]) - np.dot(x1[len(x1)/2:], x2[:len(x2)/2])) / np.linalg.norm(x1)**2
    return np.sqrt(a**2 + b**2)

def scale(points, scalingFactor):
    """ This function rescale the points by the scaling factor. """

    points = (points).dot(scalingFactor)

    return points

def getRotationAngle(template, points):
    """ This function returns the angle of rotation that best matches the points against the template. """

    numerator = sum(points[:, 0] * template[:, 1] - points[:, 1] * template[:, 0])

    divisor = sum(points[:, 0] * template[:, 0] + points[:, 1] * template[:, 1])

    #   Avoiding dividing by zero
    if divisor == 0.0:
        divisor = 0.00000000001

    return math.atan(numerator / divisor)

def rotate(points, theta, center_point=(0, 0)):
    """ This function rotates the points by theta around the center point."""

    new_array = np.array(points)

    new_array[0, :] -= center_point[0]
    new_array[1, :] -= center_point[1]

    new_array = np.dot(rotationMatrix(theta),
                    new_array.transpose()).transpose()

    new_array[0, :] += center_point[0]
    new_array[1, :] += center_point[1]

    return new_array

def rotationMatrix(theta):
    """ This function returns the rotation matrix."""

    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

def alignShapes(shape1, shape2):
    """ This function aligns two shapes. """

    theta = getRotationAngle(shape1, shape2)
    scalingFactor = getScalingFactor(shape2, shape1)

    return scale(rotate(shape2, theta), 1)

def getMeanShape(alignedShapes):
    """ This function returns the mean shape of aligned shapes."""

    return alignedShapes.mean(axis=0)

def visualizeLandmark(points, img):
        """ This function visualizes the landmark points that are in matrix notation. """

        img = np.zeros(img.shape)

        for i in range(len(points)):
            img[int(points[i, 1]), int(points[i, 0])] = 1

        cv2.imshow('Rendered shape', img)
        cv2.waitKey(0)

def visualizeLandmarkOnRadiograph(points, img):
        """ This function visualizes the landmark points that are in matrix notation on the radiograph."""

        for i in range(len(points) - 1):
            cv2.line(img, (int(points[i, 0]), int(points[i, 1])), (int(points[i+1, 0]), int(points[i+1, 1])), (255, 255, 0))
        cv2.line(img, (int(points[0, 0]), int(points[0, 1])), (int(points[len(points) - 1, 0]), int(points[len(points) - 1, 1])), (255, 255, 0))

        cv2.imshow('Rendered shape', img)
        cv2.waitKey(0)

def visualizeLandmarksOnRadiograph(landmarks, img):
        """ This function visualizes the landmarks points that are in vector notation on the radiograph."""

        for landmark in landmarks:
            points = landmarkAsMatrix(landmark)
            for i in range(len(points) - 1):
                cv2.line(img, (int(points[i, 0]), int(points[i, 1])), (int(points[i+1, 0]), int(points[i+1, 1])), (255, 255, 0))
            cv2.line(img, (int(points[0, 0]), int(points[0, 1])), (int(points[len(points) - 1, 0]), int(points[len(points) - 1, 1])), (255, 255, 0))

        points = landmarkAsMatrix(getMeanShape(landmarks))
        for i in range(len(points) - 1):
            cv2.line(img, (int(points[i, 0]), int(points[i, 1])), (int(points[i+1, 0]), int(points[i+1, 1])), (255, 0, 0))
        cv2.line(img, (int(points[0, 0]), int(points[0, 1])), (int(points[len(points) - 1, 0]), int(points[len(points) - 1, 1])), (255, 0, 0))


        cv2.imshow('Rendered shape', img)
        cv2.waitKey(0)

def GPA2(landmarks):
    """ This function performs the generalized procrustes analysis on the landmarks. """

    landmarks2 = np.zeros((landmarks.shape[0], landmarks.shape[1]/2, 2))

    # translate each landmark so its center of gravity is at the origin
    for i in range(len(landmarks)):
        landmarks2[i] = translateToOrigin(landmarkAsMatrix(landmarks[i]))

    # initial estimate of the mean
    meanShape = getMeanShape(landmarks2)

    # scale to unity
    landmarks3 = landmarks2
    for i in range(len(landmarks3)):
        landmarks2[i] = scaleTo1(landmarks3[i])

    # record x0 for default orientation
    x0 = landmarks2[0]

    while True:
        # align all shapes with the estimation of the mean shape
        landmarks3 = landmarks2
        for i in range(len(landmarks3)):
            landmarks2[i] = alignShapes(meanShape, landmarks3[i])

        # reestimate the mean from aligned shapes
        meanShape2 = getMeanShape(landmarks2)

        # apply blabla
        meanShape2 = scaleTo1(meanShape2)
        meanShape2 = alignShapes(meanShape2, x0)

        if ((landmarkAsVector(meanShape) - landmarkAsVector(meanShape2)) < 1e-10).all():

            break

        meanShape = meanShape2

    return meanShape, landmarks2

if __name__ == '__main__':
    directory = "Data/Landmarks/original"

    #visualizeLandmark(landmarkAsMatrix(getMeanShape(getLandmarks(directory, '1'))), plt.imread("Data/Radiographs/01.tif"))
    gpa = GPA2(getLandmarks(directory, '1'))
    #visualizeLandmark(gpa[0]*500+800, plt.imread("Data/Radiographs/01.tif"))
    #visualizeLandmarksOnRadiograph(getLandmarks(directory, '1'), plt.imread("Data/Radiographs/01.tif"))
    [eigenvalues, eigenvectors, mu] = pca(gpa[1], 8)
