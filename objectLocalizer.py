import math
import cv2
import numpy as np
import pickle

from imagePreprocessor import ImagePreprocessor
from dummyCamera import DummyCamera
from MapCoords import MapCoords
from filters import SizeFilter, CircularityFilter

USE_DUMMY_CAMERA = False

scaled_res = (0, 0)

im1crop = []
im2crop = []

# position of ends of camera lenses relative to origin (mm).
cam1Position = np.array([0, 48, 142])
cam2Position = np.array([0, -48, 142])

# distance between fiducials in mm
square_distance = 60

# distance face plate is above z = 0
facePlateHeight = 42

# radius of lev in mm
lev_radius = 30

# height of lev in mm
lev_height = 5

obj_min_size = 100
obj_max_size = 4000

obj_circularity = .5

class ObjectLocalizer:

    @staticmethod
    def create_filters():
        area = SizeFilter(obj_min_size, obj_max_size)
        circularity = CircularityFilter(obj_circularity)
        return [circularity, area]
        # return []

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
    # closer to z = 0 threshold should be greater because further from faceplate. Use similar triangles to find ajusted thresh
    def distance_thresh(obj_point, faceplate_point, cam_point, thresh_const):
        faceplate_distance = ObjectLocalizer.distance3d(cam_point, faceplate_point)
        obj_distance = ObjectLocalizer.distance3d(cam_point, obj_point)

        thresh = (thresh_const * obj_distance) / faceplate_distance
        return thresh

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
    def get_contours_and_apply_filters(thresh_im, filters):
        contours, hierachy = cv2.findContours(thresh_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

    # todo integrate this filter better
    @staticmethod
    def get_contour_centers_real(contours, mapcoords, zval):
        centers = []
        for contour in contours:
            px, py = ObjectLocalizer.get_contour_center(contour)
            rx, ry = mapcoords.get_real_coord(px, py)

            # filter out contours outside of lev
            dist = ObjectLocalizer.distance2d([0, 0], [rx, ry])
            print(dist)
            if dist < lev_radius:
                centers.append(np.array([rx,ry,zval]))
        print(len(centers))
        return centers

    @staticmethod
    def distance3d(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

    @staticmethod
    def distance2d(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def within_boundaries(point):
        return ObjectLocalizer.distance2d([0,0], point) and point[2] < lev_height

    def localize_object(self, frame):
        # get undistorted images
        im1, im2 = self.image_preprocessor.undistort_and_crop(frame)

        # apply blur
        im1_blur = cv2.medianBlur(im1, 5)
        im2_blur = cv2.medianBlur(im2, 5)

        # convert to grayscale
        im1_gray = cv2.cvtColor(im1_blur, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2_blur, cv2.COLOR_BGR2GRAY)

        # Loop through threshold value until object is found
        counter = self.min_thresh
        while counter <= self.max_thresh:
            ret, thresh_im1 = cv2.threshold(im1_gray, counter, 255, 0)
            ret, thresh_im2 = cv2.threshold(im2_gray, counter, 255, 0)

            contours_im1 = ObjectLocalizer.get_contours_and_apply_filters(thresh_im1, self.filters)
            contours_im2 = ObjectLocalizer.get_contours_and_apply_filters(thresh_im2, self.filters)

            # get mm coordinates of contours
            im1_cont_centers = ObjectLocalizer.get_contour_centers_real(contours_im1, self.mc1, facePlateHeight)
            im2_cont_centers = ObjectLocalizer.get_contour_centers_real(contours_im2, self.mc2, facePlateHeight)

            # todo draw centers as well

            if self.debug:
                if len(contours_im1) > 0:
                    im1_copy = np.copy(im1)
                    for c in range(len(contours_im1)):
                        cv2.drawContours(im1_copy, [contours_im1[c]], 0, (0, 0, 255), 3)

                    cv2.imwrite("/Users/andylegrand/PycharmProjects/objloc_ras_pi/thresh/" + str(counter) + "cam1.jpg",
                                im1_copy)

                if len(contours_im2) > 0:
                    im2_copy = np.copy(im2)
                    for c in range(len(contours_im2)):
                        cv2.drawContours(im2_copy, [contours_im2[c]], 0, (0, 0, 255), 3)

                    cv2.imwrite("/Users/andylegrand/PycharmProjects/objloc_ras_pi/thresh/" + str(counter) + "cam2.jpg",
                                im2_copy)

                with open("/Users/andylegrand/PycharmProjects/objloc_ras_pi/thresh/" + str(counter) + '.txt', 'w') as f:
                    f.write("contours in im1" + str(im1_cont_centers))
                    f.write("contours in im2" + str(im2_cont_centers))

            min_dist = 1000000

            # check if any contours are in range of each other
            for im1_cont_center in im1_cont_centers:
                for im2_cont_center in im2_cont_centers:
                    p1, p2 = ObjectLocalizer.closest_points_on_skew_lines(cam1Position, im1_cont_center, cam2Position, im2_cont_center)
                    # error1 = ObjectLocalizer.distance_thresh(p1, im1_cont_center, cam1Position, 3)
                    # error2 = ObjectLocalizer.distance_thresh(p2, im2_cont_center, cam2Position, 3)
                    distance = ObjectLocalizer.distance3d(p1, p2)
                    midpoint = (p1 + p2) / 2

                    if self.debug:
                        with open("/Users/andylegrand/PycharmProjects/objloc_ras_pi/thresh/" + str(counter) + '.txt', 'a') as f:
                            f.write("\n"+str(im1_cont_center) + " " + str(im2_cont_center) + " " + str(midpoint) + " " + str(distance))
                        if distance < min_dist:
                            min_dist = distance

                    if 3 > distance:  # and ObjectLocalizer.within_boundaries(midpoint):
                        return midpoint
            if self.debug:
                with open("/Users/andylegrand/PycharmProjects/objloc_ras_pi/thresh/" + str(counter) + '.txt', 'a') as f:
                    f.write("\nmindist: " + str(min_dist))

            counter += self.step
        return None

    def __init__(self, im, thresh_min, thresh_max, step, debug=False):
        self.min_thresh = thresh_min
        self.max_thresh = thresh_max
        self.step = step
        self.debug = debug

        self.filters = ObjectLocalizer.create_filters()

        # load intrensic camera parameters. Used to undistort images from cameras
        with open('cam1Params', 'rb') as f:
            cam1_params = pickle.load(f)
        with open('cam2Params', 'rb') as f:
            cam2_params = pickle.load(f)

        self.image_preprocessor = ImagePreprocessor([cam1_params, cam2_params], (1000, 1000))
        im1, im2 = self.image_preprocessor.undistort_and_crop(im)

        # initialize mapcoords
        self.mc1 = MapCoords(im1, square_distance, "/Users/andylegrand/PycharmProjects/objloc_ras_pi/output/cam1", debug=False)
        self.mc2 = MapCoords(im2, square_distance, "/Users/andylegrand/PycharmProjects/objloc_ras_pi/output/cam2", debug=False)


# debug test
def debug_test():
    frame = cv2.imread("/Users/andylegrand/PycharmProjects/objloc_ras_pi/output/testframe.jpg")
    obj_loc = ObjectLocalizer(frame, 60, 150, 5, debug=False)

    coord = obj_loc.localize_object(frame)
    print(coord)
    exit()

# demo
def demo():
    cap = cv2.VideoCapture('/Users/andylegrand/PycharmProjects/objloc_ras_pi/cardboardbackground.mp4')
    ret, frame = cap.read()

    obj_loc = ObjectLocalizer(frame, 60, 150, 5, debug=False)

    while (cap.isOpened()):
        ret, frame = cap.read()
        coord = obj_loc.localize_object(frame)

        print("coord:" + str(coord))

        text = "no object"
        if coord is not None:
            text = str(coord[0]) + " " + str(coord[1]) + " " + str(coord[2])

        # Reading an image in default mode
        image = frame

        # Window name in which image is displayed
        window_name = 'loc'

        # font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (50, 50)

        # fontScale
        fontScale = 1

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        image = cv2.putText(image, text, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)

        # Displaying the image
        cv2.imshow(window_name, image)
        cv2.waitKey(1)

if __name__ == '__main__':
    demo()

