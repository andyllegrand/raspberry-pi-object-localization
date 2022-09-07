import math
import cv2
import numpy as np
import pickle

import dirReset
from imagePreprocessor import ImagePreprocessor
from MapCoords import MapCoords
from filters import SizeFilter, CircularityFilter

scaled_res = (0, 0)

# position of ends of camera lenses relative to origin (mm). #TODO read this from pickle
cam1Position = np.array([0, 48, 142])
cam2Position = np.array([0, -48, 142])

# distance between fiducials in mm
square_distance = 75

# length of side of square
square_side_length = 5

# distance face plate is above z = 0
facePlateHeight = 43
# 45.3

# radius of lev in mm
lev_radius = 30

# calibration plate object locations
calibration_objects = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]

obj_min_size = 100
obj_max_size = 4000

obj_circularity = .5

# CRITICAL THAT THIS POINTS TO OUTPUT DIRECTORY. DIRECTORY MAY BE WIPED!!!!
output_dir = "/Users/andylegrand/PycharmProjects/objloc_ras_pi/output"
assert output_dir.split("/")[-1] == "output"


class ObjectLocalizer:

    @staticmethod
    def draw_vectors_and_faceplate(points, lines):
        """
        debug method which draws the levitator faceplate, lines, and points
        :param points: 3d points to plot
        :param lines: 3d array representing lines to plot.
            First dimension: line
            Second dimension: 2 points on the line
            Third dimension: x,y,z coordinates of each point
        :return: None
        """
        print(points)
        print(lines)
        import plotly.graph_objs as go
        import math

        fig = go.Figure()

        # draw circle
        circle_points = []
        steps = np.linspace(0, 2 * math.pi, 700)
        for rad in steps:
            circle_points.append([math.cos(rad)*lev_radius, math.sin(rad)*lev_radius, facePlateHeight])

        circle_points = np.array(circle_points)

        fig.add_trace(
            go.Scatter3d(
                x=circle_points[:, 0],
                y=circle_points[:, 1],
                z=circle_points[:, 2],
                mode='lines',
            )
        )

        # add lines
        lines = np.array(lines)
        for i, feat in enumerate(lines):
            # extend vectors
            delta = feat[0] - feat[1]
            feat[1] -= delta * .5

            fig.add_trace(
                go.Scatter3d(
                    x=feat[:, 0],
                    y=feat[:, 1],
                    z=feat[:, 2],
                    mode='lines',
                )
            )

        # add points
        points = np.array(points)
        fig.add_scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers')

        fig.show()

    @staticmethod
    def create_filters():
        """create filters used for validating contours"""
        area = SizeFilter(obj_min_size, obj_max_size)
        circularity = CircularityFilter(obj_circularity)
        return [circularity, area]

    @staticmethod
    def distance_thresh(obj_point, faceplate_point, cam_point, thresh_const):
        """
        create distance threshold which scales with the distance between the object and the camera
        :param obj_point: 3d point where object is located
        :param faceplate_point: 3d point representing location of object projected onto fiducial plane
        :param cam_point: 3d point representing cameras focal point
        :param thresh_const: acceptable amount of error for object on fiducial plane
        :return: adjusted threshold value
        """

        # use similar triangles to find adjusted threshold
        faceplate_distance = ObjectLocalizer.distance3d(cam_point, faceplate_point)
        obj_distance = ObjectLocalizer.distance3d(cam_point, obj_point)

        thresh = (thresh_const * obj_distance) / faceplate_distance
        return thresh

    @staticmethod
    def closest_points_on_skew_lines(XA0, XA1, XB0, XB1):
        """
        find the closest points on 2 lines each represented by 2 points
        :param XA0: first point on line 1
        :param XA1: second point on line 1
        :param XB0: first point on line 2
        :param XB1: second point on line 2
        :return: closest points on each line
        """
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
    def get_contour_center(contour):
        """returns the center of a given contour"""
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY

    @staticmethod
    def distance3d(p1, p2):
        """distance between 2 3d points"""
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

    @staticmethod
    def distance2d(p1, p2):
        """distance between 2 2d points"""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def within_boundaries(point):
        """check if a point is inside the lev boundaries"""
        return ObjectLocalizer.distance2d([0,0], point) <= lev_radius and 0 <= point[2] <= facePlateHeight

    @staticmethod
    def get_real_world_contour_centers(preprocessed_im, original_im, thresh, filters, mapcoords, faceplate_height):
        """
        thresholds image then finds contours. Apply filters to contour, then ensures that contours are inside of
        lev window. Draws valid contours on original image and gets the real world coordinates of the contour centers
        from mapcoords.
        :param preprocessed_im: grayscale image
        :param original_im: original image to draw found contours on
        :param thresh: thresh value to use
        :param filters: filters to apply to contours
        :param mapcoords: mapcoords object to use to find real world coordinates
        :param faceplate_height: height of faceplate relative to faceplate
        :return: real world coordinates of all valid contours
        """
        original_image_copy = np.copy(original_im)

        # get thresh image
        ret, thresh_im = cv2.threshold(preprocessed_im, thresh, 255, 0)

        # get contours
        contours, hierachy = cv2.findContours(thresh_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # apply contour filters
        contours_filter = contours
        for filter in filters:
            contours_filter = filter.apply(contours_filter)

        # get real world coordinates and filter out contours outside valid area
        centers = []
        for contour in contours_filter:
            px, py = ObjectLocalizer.get_contour_center(contour)
            rx, ry = mapcoords.get_real_coord(px, py)

            # filter out contours outside of lev
            dist = ObjectLocalizer.distance2d([0, 0], [rx, ry])
            if dist < lev_radius:
                centers.append(np.array([rx, ry, faceplate_height]))

                # draw contour on original image copy
                cv2.drawContours(original_image_copy, [contour], 0, (0, 0, 255), 3)
                cv2.circle(original_image_copy, (px, py), radius=0, color=(0, 0, 255), thickness=-1)

        return centers, original_image_copy


    @staticmethod
    def sort_points(points):
        """sorts a list of points first by their y coordinate then by their x coordinate"""
        # sort by y
        points.sort(key=lambda row: (row[1]))
        # sort by x
        points = sorted(points)
        return points

    @staticmethod
    def get_focal_point(object_points, fiducial_points):
        """
        Gets effective focal point of each camera by using calibration plate with objects at known locations and the
        locations of the projected contours.
        :param object_points: Known positions of calibration objects
        :param fiducial_points: Centers of contours of fiducial plane
        :return: focal point location
        """
        assert len(object_points) == len(fiducial_points)

        # create 2d array of points where every 2 points correspond to a line
        matched_points = []
        for p in range(len(calibration_objects)):
            matched_points.append([object_points[p], calibration_objects[p]])

        # find the closest points between every different line then average them together.
        total_points = np.array([0, 0, 0])
        for line1 in matched_points:
            for line2 in matched_points:
                if line1 is not line2:
                    point1, point2 = ObjectLocalizer.closest_points_on_skew_lines(line1[0], line1[1], line2[0],
                                                                                  line2[0])
                    midpoint = (point1 + point2) / 2
                    total_points += midpoint

        # average points
        ave = np.array(total_points) / len(calibration_objects)
        return ave

    def calibrate_focal_points(self, frame):
        """
        Finds the focal point of each camera from frame with calibration plate inserted and saves to pickle
        :param frame: raw output from cameras with calibration plate inserted inside levitator
        :return: None
        """
        static_thresh = 0 # instead of using variable thresh optimal thresh is set manually

        # get undistorted images
        im1, im2 = self.image_preprocessor.undistort_and_crop(frame)

        # apply blur
        im1_blur = cv2.medianBlur(im1, 5)
        im2_blur = cv2.medianBlur(im2, 5)

        # convert to grayscale
        im1_gray = cv2.cvtColor(im1_blur, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2_blur, cv2.COLOR_BGR2GRAY)

        im1_contour_centers, contour1_im = ObjectLocalizer.get_real_world_contour_centers(im1_gray, im1, static_thresh,
                                                                                          self.filters, self.mc1,
                                                                                          facePlateHeight)
        im2_contour_centers, contour2_im = ObjectLocalizer.get_real_world_contour_centers(im2_gray, im2, static_thresh,
                                                                                          self.filters, self.mc2,
                                                                                          facePlateHeight)
        cv2.imshow("contour1_im", contour1_im)
        cv2.imshow("contour2_im", contour2_im)
        cv2.waitKey(0)

        fp1 = ObjectLocalizer.get_focal_point(calibration_objects, im1_contour_centers)
        fp2 = ObjectLocalizer.get_focal_point(calibration_objects, im2_contour_centers)

        # pickle
        with open('cam1fp', 'wb') as f:
            pickle.dump(fp1, f)

        with open('cam2fp', 'wb') as f:
            pickle.dump(fp2, f)

        print("calibration finished")

    def localize_object(self, frame):
        """
        returns real world coordinates of object given input frame
        :param frame: input frame. Should be raw input from pi.
        :return: 3d mm coordinates of object. If no object is found returns None
        """
        # get undistorted images
        im1, im2 = self.image_preprocessor.undistort_and_crop(frame)

        # apply blur
        im1_blur = cv2.medianBlur(im1, 5)
        im2_blur = cv2.medianBlur(im2, 5)

        # convert to grayscale
        im1_gray = cv2.cvtColor(im1_blur, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2_blur, cv2.COLOR_BGR2GRAY)

        thresh_path = thresh_path = output_dir + "/thresh"

        # Loop through threshold value until object is found
        counter = self.min_thresh
        while counter <= self.max_thresh:
            im1_contour_centers, contour1_im = ObjectLocalizer.get_real_world_contour_centers(im1_gray, im1, counter, self.filters, self.mc1, facePlateHeight)
            im2_contour_centers, contour2_im = ObjectLocalizer.get_real_world_contour_centers(im2_gray, im2, counter, self.filters, self.mc2, facePlateHeight)

            if self.debug:
                cv2.imwrite(thresh_path + "/" + str(counter) + "cam1.jpg", contour1_im)
                cv2.imwrite(thresh_path + "/" + str(counter) + "cam2.jpg", contour2_im)

            # check if any contours are in range of each other
            for im1_cont_center in im1_contour_centers:
                for im2_cont_center in im2_contour_centers:
                    p1, p2 = ObjectLocalizer.closest_points_on_skew_lines(cam1Position, im1_cont_center, cam2Position, im2_cont_center)
                    error1 = ObjectLocalizer.distance_thresh(p1, im1_cont_center, cam1Position, 3)
                    error2 = ObjectLocalizer.distance_thresh(p2, im2_cont_center, cam2Position, 3)
                    distance = ObjectLocalizer.distance3d(p1, p2)
                    midpoint = (p1 + p2) / 2

                    if error1 + error2 > distance and ObjectLocalizer.within_boundaries(midpoint):
                        if self.debug:
                            ObjectLocalizer.draw_vectors_and_faceplate([im1_cont_center, im2_cont_center, midpoint], [[cam1Position, im1_cont_center], [cam2Position, im2_cont_center]])
                        return midpoint

            counter += self.step
        return None

    def __init__(self, im, thresh_min, thresh_max, step, debug=False):
        """
        create new object localizer object
        :param im: calibration image. Should be raw input from pi camera setup.
        :param thresh_min: minimum thresh value to test
        :param thresh_max: maximum thresh value to test
        :param step: value to iterate thresh by
        :param debug: If enabled will print debug messages, create mapcoords in debug mode and
        display 3d diagram for each frame
        """
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

        # reset output directory
        dirReset.reset_directory(output_dir)

        # initialize mapcoords
        self.mc1 = MapCoords(im1, square_distance, square_side_length, outputDir=output_dir+"/cam1", print_messages=True)
        self.mc2 = MapCoords(im2, square_distance, square_side_length, outputDir=output_dir+"/cam2", print_messages=True)

# debug test
def debug_test(frame):
    cal_frame = cv2.imread("/Users/andylegrand/PycharmProjects/objloc_ras_pi/test_images/testframe.jpg")
    frame = cv2.imread("/Users/andylegrand/PycharmProjects/objloc_ras_pi/test_images/testframe"+ str(frame) + ".jpg")
    obj_loc = ObjectLocalizer(cal_frame, 60, 150, 5, debug=True)

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
    # demo()
    debug_test(18)
