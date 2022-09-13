import time

import cv2
import numpy as np
from collections import defaultdict
import sys

from imagePreprocessor import ImagePreprocessor


class MapCoords:
    """
    inputs an image, locates the fiducials, then can identify the real world location of each pixel in the image on a
    2d plane. measurements are in mm and 0,0 is at the center of the 4 fiducials. It is assumed that the position of the
     camera relative to the faceplate remains static thus fiducial detection is only run one time
    """

    def print_debug(self, message):
        """only print message if debug is enabled"""
        if self.print_messages:
            print(message)

    def write_image(self, path, image):
        """only write image if debug is enabled"""
        if self.output_dir is not None:
            cv2.imwrite(self.output_dir + path, image)

    def __init__(self, image, square_distance, square_side_length, outputDir=None, print_messages=False,
                 show_cropped_fiducials=False):
        """
        initialize new mapCoords object.

        :param image: reference image where fiducials are found
        :param square_distance: distance between the top left corners of two adjacent fiducials
        :param square_side_length: the side length of a fiducial square
        :param outputDir: directory to write output images to. Set to None if no images should be written.
        :param print_messages: set to true for debug print messages
        :param show_cropped_fiducials: if set to true show the outline of each fiducial. Displayed during
            initialization
        """
        self.print_messages = print_messages
        self.show_cropped_fiducials = show_cropped_fiducials
        self.image = image
        self.cm_distance = square_distance
        self.output_dir = outputDir
        self.square_side_length = square_side_length
        src_points = []
        dst_points = []

        # constants for building dst points
        offset_distance = square_distance
        fiducial_top_left_points = [[0, 0], [0, offset_distance], [offset_distance, offset_distance],
                                    [offset_distance, 0]]
        fiducial_offsets = [[0, 0], [0, square_side_length], [square_side_length, square_side_length],
                            [square_side_length, 0]]

        # find upper left and lower right coordinates of each quadrant of the image
        h, w, c = np.shape(image)
        center = [int(h / 2), int(w / 2)]
        expected_positions_left = [[0, 0], [center[0], 0], [center[1], center[0]], [0, center[0]]]
        expected_positions_right = [[center[0], center[1]], [h, center[1]], [h, w], [center[1], w]]

        self.print_debug(expected_positions_left)
        self.print_debug(expected_positions_right)

        # crop each fiducial, identify all 4 corners, then save corner closest to center of image
        for i in range(4):
            self.print_debug("processing fiducial: " + str(i) + "...")
            cropped_fiducial = MapCoords.crop(image, expected_positions_left[i], expected_positions_right[i])
            mask = MapCoords.preprocess_image(cropped_fiducial)
            edges = MapCoords.find_outline(mask)

            if self.show_cropped_fiducials:
                cv2.imshow("cf", cropped_fiducial)
                cv2.imshow("mask", mask)
                cv2.imshow("edges", edges)
                cv2.waitKey(0)

            # try different houghline threshold values until 4 corners are found
            thresh = 30
            while True:
                if thresh <= 0:
                    cv2.imshow("cf", cropped_fiducial)
                    cv2.waitKey(0)
                    exit()

                lines, line_img = MapCoords.draw_vertical_horizontal_lines(edges, thresh, cropped_fiducial)
                self.write_image("/fiducialHoughLines" + str(i) + "thresh" + str(thresh) + ".jpg", line_img)
                corners = MapCoords.find_and_group_intersections(lines)
                if len(corners) == 4:
                    break
                thresh -= 1

            # plot corners
            self.write_image("/4Corners" + str(i) + ".jpg", MapCoords.plot_corners(cropped_fiducial, corners))

            # order points
            # 1     2
            # 4     3
            MapCoords.sort_points(corners)

            # offset all coordinates to full image position and add to src points, also build dst_points array
            for c in range(4):
                src_points.append([corners[c][0] + expected_positions_left[i][0], corners[c][1] +
                                   expected_positions_left[i][1]])
                dst_points.append([fiducial_top_left_points[i][0] + fiducial_offsets[c][0],
                                   fiducial_top_left_points[i][1] + fiducial_offsets[c][1]])
            self.print_debug("done")

        # write image with all fiducial corners identified
        self.write_image("/allCorners.jpg", MapCoords.plot_corners(image, src_points))
        # find homography
        self.homography_transform = cv2.findHomography(np.array(src_points), np.array(dst_points))[0]

    def get_real_coord(self, px, py):
        """
        returns real world coordinate of a given xy pixel value
        :param px: pixel x value
        :param py: pixel y value
        :return: real world coordinate of pixel value
        """
        return MapCoords.apply_homography_transform(self.homography_transform, px, py) \
               - np.array([(self.cm_distance+self.square_side_length)/2, (self.cm_distance+self.square_side_length)/2])
        # subtract half of total side length to place 0,0 in the center

    def get_image(self):
        """
        accessor method for reference image
        :return: reference image used during initialization
        """
        return self.image

    def reconstruct_image(self):
        """
        debugging method which reconstructs the reference image based on the real world coordinates of each pixel.
        :return: Reconstructed image
        """
        scale = 10
        output = np.ones([self.cm_distance*scale+5, self.cm_distance*scale+5, 3])
        ySize, xSize, channels = self.image.shape

        for x in range(ySize):
            for y in range(xSize):
                rval = self.get_real_coord(x, y)
                if abs(rval[0]) < 30 and abs(rval[1]) < 30:
                    temp = rval + np.array([30, 30])
                    outputX = int(temp[0] * scale)-1
                    outputY = int(temp[1] * scale) - 1
                    output[outputY][outputX] = self.image[y][x]

        return output

    @staticmethod
    def preprocess_image(image):
        """blur and threshold image"""
        image_copy = np.copy(image)

        # apply blur
        image_copy = cv2.medianBlur(image_copy, 5)

        # Convert to HSV
        img_hsv = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)

        # Threshold (have to threshold twice and combine because red hue value wraps)
        thresh = cv2.inRange(img_hsv, (0, 200, 50), (10, 255, 255)) + cv2.inRange(img_hsv, (170, 200, 50), (180, 255, 255))

        return thresh

    @staticmethod
    def crop(img, point1, point2):
        """
        crops image
        :param img: image to be cropped
        :param point1: upper left point of crop (column,row)
        :param point2: lower right point to crop (column,row)
        :return: cropped image
        """
        cropped_im = img[point1[1]:point2[1], point1[0]:point2[0]]
        return cropped_im

    @staticmethod
    def find_outline(img):
        """find outlines on threshold image"""
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S

        # Sobel filter version
        grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        edges = cv2.addWeighted(abs_grad_x, 1, abs_grad_y, 1, 0)

        return edges

    # have to change thresh for each resolution scalar
    @staticmethod
    def draw_vertical_horizontal_lines(edge_image, thresh, original_image, k=2, **kwargs):
        """draw houghlines then sort into horizontal and vertical groups"""
        rho, theta = 1, np.pi / 180
        lines = cv2.HoughLines(edge_image, rho, theta, thresh)
        if lines is None:
            print("no lines")
            return None

        img = original_image.copy()
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 10000 * (-b))
            y1 = int(y0 + 10000 * (a))
            x2 = int(x0 - 10000 * (-b))
            y2 = int(y0 - 10000 * (a))

            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Set parameters
        default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
        criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
        flags = kwargs.get('flag', cv2.KMEANS_RANDOM_CENTERS)
        attempts = kwargs.get('attempts', 10)

        # Get the angles in radians
        angles = np.array([line[0][1] for line in lines])
        # Get unit circle coordinates of the angle
        pts = np.array([[np.cos(2 * angle), np.sin(2 * angle)] for angle in angles], dtype=np.float32)

        # Group the points based on where they are on the unit circle
        labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
        labels = labels.reshape(-1)

        # Group the lines based on the kmeans grouping
        segmented = defaultdict(list)
        for i, line in enumerate(lines):
            segmented[labels[i]].append(line)
        segmented = list(segmented.values())

        return segmented, img

    @staticmethod
    def find_and_group_intersections(lines):
        """given grouped lines find all intersections between groups. Then average intersections which are close to
            each other"""
        intersections = []

        # make sure that there are lines to work with
        if lines is None:
            return intersections

        # Compare lines in each group
        for i, group in enumerate(lines[:-1]):
            for next_group in lines[i + 1:]:
                for line1 in group:
                    for line2 in next_group:
                        intersections.append(MapCoords.intersection(line1, line2))

        # Average the points that are very close to each other
        # because they are likely noise
        filtered = []
        bitmap = [1 for i in range(len(intersections))]
        i = 0
        tolerance = 15
        while i < len(intersections):
            if bitmap[i] == 1:
                similar = []
                similar_idx = []
                this_point = intersections[i]
                j = i + 1
                while j < len(intersections):
                    other_point = intersections[j]
                    if np.linalg.norm(np.array(this_point) - np.array(other_point)) < tolerance:
                        similar.append(other_point)
                        similar_idx.append(j)
                    j += 1
                if len(similar) > 0:
                    similar.append(this_point)
                    avg = np.mean(similar, axis=0)
                    processed = avg.tolist()
                    processed[0] = int(np.round(processed[0]))
                    processed[1] = int(np.round(processed[1]))
                    filtered.append(processed)
                    for idx in similar_idx:
                        bitmap[idx] = 0
                else:
                    filtered.append(this_point)
            i += 1
        return filtered

    @staticmethod
    def intersection(line1, line2):
        """find intersection between 2 lines"""
        # Find intersection point of two lines from rho and theta
        # Solve:
        # x * cos(theta1) + y * sin(theta1) = r1
        # x * cos(theta2) + y * sin(theta2) = r1

        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        return [int(np.round(x0)), int(np.round(y0))]

    @staticmethod
    def plot_corners(img, points):
        """draw all points on image"""
        copy = img.copy()
        for point in points:
            copy = cv2.circle(copy, (point[0], point[1]), radius=5, color=(255, 0, 0), thickness=-1)
            cv2.putText(copy, "(x,y): " + str(point[0]) + ", " + str(point[1]),
                        (point[0], point[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        return copy

    @staticmethod
    def sort_points(points):
        """sort list of points in place in following order
            1       2
            4       3

        """
        # sort by y
        points.sort(key=lambda row: (row[1]))
        # break into 2 arrays then sort by x
        points[0:2] = sorted(points[0:2])
        points[2:4] = sorted(points[2:4], reverse=True)

    @staticmethod
    def apply_homography_transform(M, x: int, y: int):
        """applies homography transform to 1 pixel"""
        d = M[2][0] * x + M[2][1] * y + M[2][2]

        return np.array(
            [(M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d, (M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d]
        )


class visual_Test:
    """
    class for testing mapcoords.

    Reference image of the mapcoords class is displayed, then the user can click any pixel to view its real world
    coordinates.
    """
    def __init__(self, mc):
        # reading the image
        self.img = mc.get_image()
        self.ed = mc

        # displaying the image
        cv2.imshow('image', self.img)

        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image', self.click_event)

        # wait for a key to be pressed to exit
        cv2.waitKey(0)

        # close the window
        cv2.destroyAllWindows()

    def click_event(self, event, x, y, flags, params):
        print(str(x)+" "+str(y))
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:

            realCoord = self.ed.get_real_coord(x, y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(self.img, str(round(realCoord[0],2)) + ',' +
                        str(round(realCoord[1],2)), (x, y), font,
                        1, (255, 0, 255), 2)

            cv2.imshow('image', self.img)
        time.sleep(.1)

        # checking for right mouse clicks
        if event == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = self.img[y, x, 0]
            g = self.img[y, x, 1]
            r = self.img[y, x, 2]
            cv2.putText(self.img, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x, y), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image', self.img)

# testing script which creates a mapcoords object, then runs the reconstruct_image and visual_test tests
if __name__ == '__main__':
    import pickle

    im = cv2.imread("/Users/andylegrand/PycharmProjects/objloc_ras_pi/test_images/testframe.jpg")

    with open('cam1Params', 'rb') as f:
        cam1_params = pickle.load(f)
    with open('cam2Params', 'rb') as f:
        cam2_params = pickle.load(f)

    image_preprocessor = ImagePreprocessor([cam1_params, cam2_params], (1000, 1000))
    im1, im2 = image_preprocessor.undistort_and_crop(im)


    mc1 = MapCoords(im1, 60, 5, outputDir="/Users/andylegrand/PycharmProjects/objloc_ras_pi/output/cam1", show_cropped_fiducials=True)
    cv2.imwrite(mc1.output_dir+"/recon.jpg",mc1.reconstruct_image())
    visual_Test(mc1)