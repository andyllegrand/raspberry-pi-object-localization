import math
import cv2
import numpy as np
import pickle

from cameraClass import Camera
from dummyCamera import DummyCamera
from MapCoords import MapCoords

USE_DUMMY_CAMERA = False

scaled_res = (0, 0)

im1crop = []
im2crop = []

# position of ends of camera lenses relative to origin (mm).
cam1Position = [0, 48, 142]
cam2Position = [0, -48, 142]

# distance between fiducials in mm
square_distance = 60

# distance face plate is above z = 0
facePlateHeight = 42


class ObjectLocalizer:

    # following 2 methods from https://stackoverflow.com/questions/55641425/check-if-two-contours-intersect
    @staticmethod
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    @staticmethod
    def contour_intersect(cnt_ref, cnt_query):

        ## Contour is a list of points
        ## Connect each point to the following point to get a line
        ## If any of the lines intersect, then break

        for ref_idx in range(len(cnt_ref) - 1):
            ## Create reference line_ref with point AB
            A = cnt_ref[ref_idx][0]
            B = cnt_ref[ref_idx + 1][0]

            for query_idx in range(len(cnt_query) - 1):
                ## Create query line_query with point CD
                C = cnt_query[query_idx][0]
                D = cnt_query[query_idx + 1][0]

                ## Check if line intersect
                if ObjectLocalizer.ccw(A, C, D) != ObjectLocalizer.ccw(B, C, D) and ObjectLocalizer.ccw(A, B, C) != ObjectLocalizer.ccw(A, B, D):
                    ## If true, break loop earlier
                    return True

        return False

    @staticmethod
    # closer to z = 0 threshold should be greater because further from faceplate
    def distance_thresh(obj_point, cam_point, z_height, thresh_const):

        return 0

    @staticmethod
    def closest_points_on_skew_lines(XA0, XA1, XB0, XB1):
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

        return point1, point2

    @staticmethod
    def get_countours_and_apply_filters(im, thresh_val, filters):
        thresh_im = cv2.threshold(im, thresh_val, 255, 0)
        contours, hierachy = cv2.findContours(thresh_val, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours_filter = contours
        for filter in filters:
            contours_filter = filter.apply(contours_filter)

        return contours_filter

    @staticmethod
    def get_contour_center(contour):
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY

    @staticmethod
    def get_contour_centers_real(contours, mapcoords, zval):
        centers = []
        for contour in contours:
            px, py = ObjectLocalizer.get_contour_center(contour)
            rx, ry = mapcoords.get_real_coord(px, py)
            centers.append(np.array([rx,ry,zval]))
        return centers

    @staticmethod
    def distance3d(p1, p2):
        return math.sqrt((p1[0]+p2[0])**2 + (p1[1]+p2[1])**2 + (p1[2]+p2[2])**2)

    @staticmethod
    def within_boundaries(point):
        return True

    def localize_object(self):
        # get undistorted images
        im1, im2 = None, None

        im1_blur = cv2.medianBlur(im1, 5)
        im2_blur = cv2.medianBlur(im2, 5)

        im1_gray = cv2.cvtColor(im1_blur, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2_blur, cv2.COLOR_BGR2GRAY)

        counter = self.min_thresh
        while counter <= self.max_thresh:
            contours_im1 = ObjectLocalizer.get_countours_and_apply_filters(im1_gray, counter, self.filters)
            contours_im2 = ObjectLocalizer.get_countours_and_apply_filters(im2_gray, counter, self.filters)

            im1_cont_centers = ObjectLocalizer.get_contour_centers_real(contours_im1, self.mc1, facePlateHeight)
            im2_cont_centers = ObjectLocalizer.get_contour_center_real(contours_im2, self.mc2, facePlateHeight)

            for im1_cont_center in im1_cont_centers:
                for im2_cont_center in im2_cont_centers:
                    p1, p2 = ObjectLocalizer.closest_points_on_skew_lines(cam1Position, im1_cont_center, cam2Position, im2_cont_center)
                    ave_z = (p1[2] + p2[2])/2
                    thresh = ObjectLocalizer.distance_thresh(ave_z)
                    distance = ObjectLocalizer.distance3d(p1, p2)
                    midpoint = (p1 + p2) / 2
                    if thresh > distance and ObjectLocalizer.within_boundaries(midpoint):
                        return midpoint

            counter += self.step

        return None

    def __init__(self, min, max, step):
        self.min_thresh = min
        self.max_thresh = max
        self.step = step

        if not USE_DUMMY_CAMERA:
            # load intrensic camera parameters. Used to undistort images from cameras
            with open('cam1Params', 'rb') as f:
                cam1_params = pickle.load(f)
            with open('cam2Params', 'rb') as f:
                cam2_params = pickle.load(f)

            self.camera = Camera([cam1_params, cam2_params], scaled_res, im1crop, im2crop)
        else:
            self.camera = DummyCamera(None, scaled_res, im1crop, im2crop)
        # initialize camera

        # initialize mapcoords
        self.mc1 = MapCoords()
        self.mc2 = MapCoords()
        return