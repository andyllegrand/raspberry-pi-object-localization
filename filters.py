import math

import cv2

# use similar triangles to find min and max pixel areas of object
def find_min_max_areas(radius, picture_res, cam_pos, lev_top, faceplate_height, mm_distance):
    allowed_deviation_factor = 0

    # at faceplate height:
    real_area_fp = mm_distance**2
    full_res_fp = picture_res[0] * picture_res[1]

    # top of lev:



    return 0,0

class SizeFilter:
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def apply(self, contours):
        passed = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min < area < self.max:
                passed.append(contour)
        return passed

class CircularityFilter:
    def __init__(self, circularity):
        self.circularity_thresh = circularity

    def apply(self, contours):
        passed = []
        for contour in contours:
            area = cv2.contourArea(contour)
            arclength = cv2.arcLength(contour, True)
            if arclength != 0.0:
                circularity = 4 * math.pi * area / (arclength * arclength)
                if self.circularity_thresh < circularity:
                    passed.append(contour)
        return passed

