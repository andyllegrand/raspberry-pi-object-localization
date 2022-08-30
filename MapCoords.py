import time

import cv2
import numpy as np
from collections import defaultdict
import sys

from imagePreprocessor import ImagePreprocessor


class MapCoords:
    RED_THRESH = 125
    GREEN_BLUE_THRESH = 60

    def print_debug(self, message):
        if self.print_messages:
            print(message)

    def write_image(self, path, image):
        if self.output_dir is not None:
            cv2.imwrite(self.output_dir + path, image)

    def __init__(self, image, cm_distance, square_side_length, outputDir=None, print_messages=False, show_cropped_fiducials=False):
        self.print_messages = print_messages
        self.show_cropped_fiducials = show_cropped_fiducials
        self.image = image
        self.cm_distance = cm_distance
        self.output_dir = outputDir
        src_points = []
        dst_points = []

        # get upper left and lower right coordinates of quadrants
        h, w, c = np.shape(image)
        center = [int(h / 2), int(w / 2)]

        expected_positions_left = [[0, 0], [center[0], 0], [center[1], center[0]], [0, center[0]]]
        expected_positions_right = [[center[0], center[1]], [h, center[1]], [h, w], [center[1], w]]

        self.print_debug(expected_positions_left)
        self.print_debug(expected_positions_right)

        # constants for building dst points
        offset_distance = cm_distance - square_side_length
        fiducial_top_left_points = [[0, 0], [0, offset_distance], [offset_distance, offset_distance],
                                    [offset_distance, 0]]
        fiducial_offsets = [[0, 0], [0, square_side_length], [square_side_length, square_side_length],
                            [square_side_length, 0]]

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
                src_points.append([corners[c][0] + expected_positions_left[i][0], corners[c][1] + expected_positions_left[i][1]])
                dst_points.append([fiducial_top_left_points[i][0] + fiducial_offsets[c][0], fiducial_top_left_points[i][1] + fiducial_offsets[c][1]])
            self.print_debug("done")

        # print images
        self.write_image("/allCorners.jpg", MapCoords.plot_corners(image, src_points))
        # find homography
        self.homography_transform = cv2.findHomography(np.array(src_points), np.array(dst_points))[0]

    def get_real_coord(self, rx, ry):
        return MapCoords.warp_point(self.homography_transform, rx, ry) - np.array([30,30]) # subtract 30 to set 0,0 to center

    def get_image(self):
        return self.image

    def reconstruct_image(self):
        scale = 10
        output = np.ones([self.cm_distance*scale+5, self.cm_distance*scale+5, 3])
        ySize, xSize, channels = self.image.shape
        print(str(xSize) + " " + str(ySize))

        for x in range(ySize):
            for y in range(xSize):
                rval = self.get_real_coord(x, y)
                if abs(rval[0]) < 30 and abs(rval[1]) < 30:
                    temp = rval + np.array([30, 30])
                    outputX = int(temp[0] * scale)-1
                    outputY = int(temp[1] * scale) - 1
                    output[outputY][outputX] = self.image[y][x]
        print("writing image")
        cv2.imwrite('/Users/andylegrand/PycharmProjects/objloc_ras_pi/output/reim.jpg', output)

    @staticmethod
    def preprocess_image(image):
        image_copy = np.copy(image)

        # apply blur
        image_copy = cv2.medianBlur(image_copy, 5)

        # split into BGR
        B, G, R = cv2.split(image_copy)

        # filter out red squares by looking for high red and low green and blue
        ret, B = cv2.threshold(B, MapCoords.GREEN_BLUE_THRESH, 80, cv2.THRESH_BINARY_INV)
        ret, G = cv2.threshold(G, MapCoords.GREEN_BLUE_THRESH, 80, cv2.THRESH_BINARY_INV)
        ret, R = cv2.threshold(R, MapCoords.RED_THRESH, 80, cv2.THRESH_BINARY)

        # must pass threshold in all color channels
        combined = B + G + R
        ret, combined = cv2.threshold(combined, 200, 255, cv2.THRESH_BINARY)

        return combined

    @staticmethod
    def crop(img, point1, point2):
        cropped_im = img[point1[1]:point2[1], point1[0]:point2[0]]
        return cropped_im

    @staticmethod
    def find_outline(img):
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
        copy = img.copy()
        for point in points:
            copy = cv2.circle(copy, (point[0], point[1]), radius=5, color=(255, 0, 0), thickness=-1)
            cv2.putText(copy, "(x,y): " + str(point[0]) + ", " + str(point[1]),
                        (point[0], point[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        return copy

    @staticmethod
    def calc_distance(point1, point2):
        return np.sqrt((point1[1]-point2[1]) ** 2 + (point1[0]-point2[0]) ** 2)

    @staticmethod
    def find_closest_corner(points, ref):
        minDist = sys.float_info.max
        closest_point = None
        for point in points:
            dist = MapCoords.calc_distance(ref, point)
            if MapCoords.calc_distance(ref, point) < minDist:
                closest_point = point
                minDist = dist
        return closest_point

    # sorts in place
    # 1     2
    # 3     4
    @staticmethod
    def sort_points(points):
        # sort by y
        points.sort(key=lambda row: (row[1]))
        # break into 2 arrays then sort by x
        points[0:2] = sorted(points[0:2])
        points[2:4] = sorted(points[2:4], reverse=True)

    @staticmethod
    def average_points(points):
        total = [0,0]
        for point in points:
            total[0] += point[0]
            total[1] += point[1]
        return[total[0] / len(points), total[1] / len(points)]

    @staticmethod
    def eval_linear_equation(m, x, b):
        return m*x+b

    # Ax + By + C = 0
    @staticmethod
    def calc_distance_between_line_and_point(point, A, B, C):
        return np.abs((point[0]*A+point[1]*B+C)/np.sqrt(A**2+B**2))

    @staticmethod
    def calc_slope(point1, point2):
        return (point1[1] - point2[1]) / (point1[0] - point2[0])

    @staticmethod
    def calc_slope_y(point1, point2):
        return (point1[0] - point2[0]) / (point1[1] - point2[1])

    @staticmethod
    def warp_point(M, x: int, y: int):
        d = M[2][0] * x + M[2][1] * y + M[2][2]

        return np.array(
            [(M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d, (M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d]
        )

    @staticmethod
    def get_homography_transform(reference_points, distance):
        src_points = np.array(reference_points)
        dst_points = np.array([[0,0],[0,distance],[distance,distance], [distance,0]])

        # find constants with cv2 method
        transform = cv2.findHomography(src_points, dst_points)

        return transform

    @staticmethod
    def fill_position_matrix_tilted(image, reference_points, distance):
        height, width, channels = image.shape
        position_matrix = -1 * np.ones((width, height), dtype=object)

        # get 4 inner corners
        top_left = reference_points[0]
        top_right = reference_points[1]
        bottom_left = reference_points[2]
        bottom_right = reference_points[3]

        # calculate side equations (start at top bc y grows down)
        left_m = MapCoords.calc_slope_y(top_left, bottom_left)
        left_b = top_left[0]
        right_m = MapCoords.calc_slope_y(top_right, bottom_right)
        right_b = top_right[0]

        # calculate top equation
        top_m = MapCoords.calc_slope(top_left, top_right)
        top_b = top_left[0]

        vertical_distance = MapCoords.calc_distance(top_left, bottom_left)

        for yc in range(bottom_left[1]-top_left[1]+1):
            left_point = [MapCoords.eval_linear_equation(left_m, yc, left_b), yc + top_left[1]]
            right_point = [MapCoords.eval_linear_equation(right_m, yc, right_b), yc + top_right[1]]

            real_y = distance * MapCoords.calc_distance_between_line_and_point(left_point, top_m, -1,
                top_left[1]) / vertical_distance

            lr_distance = MapCoords.calc_distance(left_point, right_point)
            for xc in range(int(right_point[0]-left_point[0]+1)):
                pixel = [int(left_point[0] + xc), int(MapCoords.eval_linear_equation(top_m, xc, left_point[1]))]
                real_x = distance * MapCoords.calc_distance(pixel, left_point) / lr_distance
                position_matrix[pixel[0], pixel[1]] = [real_x, real_y]
        return position_matrix


# testing class. Shows image then user can click on any pixel to get real world coordinates
class visual_Test:
    def __init__(self, ed):
        # reading the image
        self.img = ed.get_image()
        self.ed = ed

        # displaying the image
        cv2.imshow('image', self.img)

        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image', self.click_event)

        # wait for a key to be pressed to exit
        cv2.waitKey(0)

        # close the window
        cv2.destroyAllWindows()


    # function to display the coordinates of
    # of the points clicked on the image
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


if __name__ == '__main__':
    import pickle

    im = cv2.imread("/test_images/testframe.jpg")

    # load intrensic camera parameters. Used to undistort images from cameras
    with open('cam1Params', 'rb') as f:
        cam1_params = pickle.load(f)
    with open('cam2Params', 'rb') as f:
        cam2_params = pickle.load(f)

    image_preprocessor = ImagePreprocessor([cam1_params, cam2_params], (1000, 1000))
    im1, im2 = image_preprocessor.undistort_and_crop(im)

    # initialize mapcoords
    mc1 = MapCoords(im1, 60, 5, outputDir="/Users/andylegrand/PycharmProjects/objloc_ras_pi/output/cam1", show_cropped_fiducials=True)
    visual_Test(mc1)