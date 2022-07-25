import cv2
import pickle

import numpy as np

from cameraClass import Camera
from MapCoords import MapCoords

from objectDetector import objectDetector

# expected pixel values for 4 inner corners of fiducial squares.
expectedLeftCornersCam1 = []
expectedLeftCornersCam2 = []

# expected pixel values for 4 outer corners of fiducial squares.
expectedRightCornersCam1 = []
expectedRightCornersCam2 = []

# position of ends of camera lenses.
cam1Position = []
cam2Position = []

# distance between fiducials in mm
square_distance = 60

# distance face plate is above z = 0
facePlateHeight = 40

def closest_point_between_skew_lines(XA0, XA1, XB0, XB1):
    # compute unit vectors of directions of lines A and B
    UA = (XA1 - XA0) / np.linalg.norm(XA1 - XA0)
    UB = (XB1 - XB0) / np.linalg.norm(XB1 - XB0)
    # find unit direction vector for line C, which is perpendicular to lines A and B
    UC = np.cross(UB, UA)
    UC /= np.linalg.norm(UC)

    # solve the system derived in user2255770's answer from StackExchange: https://math.stackexchange.com/q/1993990
    RHS = XB0 - XA0
    LHS = np.array([UA, -UB, UC]).T
    t1, t2, t3 = np.linalg.solve(LHS, RHS)

    point1 = XA0 + t1 * UA
    point2 = XB0 + t2 * UB

    avg = (point1 + point2) / 2
    return avg

class objectLocalizer:

    def __init__(self):
        # load intrensic camera parameters. Used to undistort images from cameras
        with open('camParams1.obj', 'r') as f:
            cam1_params = pickle.load(f)
        with open('camParams1.obj', 'r') as f:
            cam2_params = pickle.load(f)

        self.camera = Camera([cam1_params, cam2_params])

        # load last used pictures
        self.el1 = cv2.imread('emptyLev1')
        self.el2 = cv2.imread('emptyLev2')

        # create edgeDetectors
        self.edgeDetector1 = MapCoords(el1, expectedLeftCornersCam1, expectedLeftCornersCam2, square_distance)
        self.edgeDetector2 = MapCoords(el2, expectedLeftCornersCam2, expectedLeftCornersCam2, square_distance)

        # initialize object detector
        self.objectDetector = objectDetector()

    def recalibrate(self):
        # get empty images
        el1, el2 = self.camera.takePic()

        cv2.imwrite('emptyLev1', el1)
        cv2.imwrite('emptyLev2', el2)

        # create new edgeDetectors
        self.edgeDetector1 = MapCoords(el1, expectedLeftCornersCam1, expectedLeftCornersCam2, square_distance)
        self.edgeDetector2 = MapCoords(el2, expectedLeftCornersCam2, expectedLeftCornersCam2, square_distance)

    def localizeObject(self):
        # get images
        img1, img2 = self.camera.takePic

        # find object pixel values
        obj1pv = self.objectDetector.detectObject(self.el1, img1)
        obj2pv = self.objectDetector.detectObject(self.el2, img2)

        # find real world coordinates (projected onto fiducial plane)
        xyCam1 = self.edgeDetector1.get_real_coord(obj1pv[0], obj1pv[1])
        xyCam2 = self.edgeDetector2.get_real_coord(obj2pv[0], obj2pv[1])

        # create vectors between camera and feducial plane coordinates.
        XA0 = np.array(xyCam1.append(facePlateHeight))
        XA1 = np.array(cam1Position)
        XB0 = np.array(xyCam2.append(facePlateHeight))
        XB1 = np.array(cam2Position)

        return closest_point_between_skew_lines(XA0, XA1, XB0, XB1)
