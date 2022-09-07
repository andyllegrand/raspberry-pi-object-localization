import math

import cv2

"""filters for contours. Each should have an apply method which inputs a list of contours and returns a list of 
contours which passed the filter"""


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



