import cv2
import pickle

import numpy
import numpy as np

from cameraClass import Camera
from MapCoords import MapCoords

from objectDetector import objectDetector

DEBUG = True

full_res = (4056, 3040)

# inverse resolution scalar. Less means more accuracy, less means better runtime and less memory issues
res_scaler = 4
x = int(full_res[0] / res_scaler)
y = int(full_res[1] / res_scaler)

# expected pixel values for 4 inner corners of fiducial squares (in full res).
expectedLeftCornersCam1 = np.array([[1500, 950], [2700,950], [1300,2250], [2900,2250]])
expectedLeftCornersCam2 = np.array([[1400, 1100], [3000,1120], [1500,2450], [2750,2480]])
expectedLeftCornersCam1 = (expectedLeftCornersCam1/res_scaler).astype(int)
expectedLeftCornersCam2 = (expectedLeftCornersCam2/res_scaler).astype(int)

# expected pixel values for 4 outer corners of fiducial squares (in full res).
expectedRightCornersCam1 = np.array([[2000, 1450], [3200,1450], [1800,2750], [3400,2750]])
expectedRightCornersCam2 = np.array([[1900, 1600], [3500,1620], [2000,2950], [3150,2980]])
expectedRightCornersCam1 = (expectedRightCornersCam1/res_scaler).astype(int)
expectedRightCornersCam2 = (expectedRightCornersCam2/res_scaler).astype(int)

print(expectedLeftCornersCam2)
print(expectedRightCornersCam2)

# position of ends of camera lenses relative to origin (mm).
cam1Position = [0, 48, 132]
cam2Position = [0, -48,132]

# distance between fiducials in mm
square_distance = 60

# distance face plate is above z = 0
facePlateHeight = 40


def print_debug(message):
    if DEBUG:
        print(message)

def imwrite_debug(path, image):
    if DEBUG:
        cv2.imwrite(path, image)

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

    # recalibrate: determines whether to create edge detectors from last used images. If set to false recalibrate
    # method is called
    def __init__(self, recalibrate=True):
        # load intrensic camera parameters. Used to undistort images from cameras
        with open('cam1Params', 'rb') as f:
            cam1_params = pickle.load(f)
        with open('cam2Params', 'rb') as f:
            cam2_params = pickle.load(f)

        cam2_params = cam1_params # TODO find a better solution for this
        print_debug("starting cameras...")
        self.camera = Camera([cam1_params, cam2_params], res_scaler)

        self.edgeDetector1 = None
        self.edgeDetector2 = None
        if recalibrate:
            self.recalibrate()
        else:
            # load last used pictures
            print_debug("loading previous images...")
            el1 = cv2.imread('/home/pi/piObjLocSync/emptyLev1.jpg')
            self.el1 = cv2.resize(el1, (x, y))
            el2 = cv2.imread('/home/pi/piObjLocSync/emptyLev2.jpg')
            self.el2 = cv2.resize(el2, (x, y))
            assert self.el1 is not None and self.el1 is not None, "images not read in properly"
            print_debug("done")

            cv2.imshow("yeet", el2)

            # create edgeDetectors
            print_debug("creating edge detectors...")
            self.edgeDetector1 = MapCoords(self.el1, res_scaler, expectedLeftCornersCam1, expectedRightCornersCam1,
                                           square_distance, "/home/pi/piObjLocSync/output/cam1")
            self.edgeDetector2 = MapCoords(self.el2, res_scaler, expectedLeftCornersCam2, expectedRightCornersCam2,
                                           square_distance, "/home/pi/piObjLocSync/output/cam2")
            print_debug("done")

        # initialize object detector
        print_debug("creating object detector...")
        self.objectDetector = objectDetector()
        print_debug("done")
        print_debug("initialization complete")

    def recalibrate(self):
        # get empty images
        self.el1, self.el2 = self.camera.take_pic()

        assert self.el1 is not None and self.el2 is not None

        cv2.imwrite('/home/pi/piObjLocSync/emptyLev1.jpg', self.el1)
        cv2.imwrite('/home/pi/piObjLocSync/emptyLev2.jpg', self.el2)

        # create new edgeDetectors
        self.edgeDetector1 = MapCoords(self.el1, res_scaler, expectedLeftCornersCam1, expectedRightCornersCam1, square_distance, "/home/pi/piObjLocSync/output/cam1")
        self.edgeDetector2 = MapCoords(self.el2, res_scaler, expectedLeftCornersCam2, expectedRightCornersCam2, square_distance, "/home/pi/piObjLocSync/output/cam2")

    def takePicturesAndLocalizeobject(self):
        im1, im2 = self.camera.take_pic()
        self.localizeObject(im1, im2)

    def localizeObject(self, img1, img2):
        # find object pixel values
        obj1pv = self.objectDetector.detectObject(self.el1, img1)
        obj2pv = self.objectDetector.detectObject(self.el2, img2)

        # find real world coordinates (projected onto fiducial plane)
        xyCam1 = self.edgeDetector1.get_real_coord(obj1pv[0], obj1pv[1])
        xyCam2 = self.edgeDetector2.get_real_coord(obj2pv[0], obj2pv[1])

        # create vectors between camera and fiducial plane coordinates.
        XA0 = np.array(xyCam1.append(facePlateHeight))
        XA1 = np.array(cam1Position)
        XB0 = np.array(xyCam2.append(facePlateHeight))
        XB1 = np.array(cam2Position)

        return closest_point_between_skew_lines(XA0, XA1, XB0, XB1)

if __name__ == '__main__':
    ol = objectLocalizer()
    input("enter to go")
    ol.takePicturesAndLocalizeobject()
