import cv2
import numpy as np
from collections import defaultdict
import sys


class MapCoords:
    RED_THRESH = 100
    GREEN_BLUE_THRESH = 50

    output_dir = "/Users/andylegrand/PycharmProjects/localization_image_testing/output"

    def __init__(self, image, expected_positions_left, expected_positions_right, cm_distance):
        self.image = image
        self.cm_distance = cm_distance
        inner_corners = []
        l, w, c = np.shape(image)
        center = [int(l/2), int(w/2)]

        # crop each fiducial, identify all 4 corners, then save corner closest to center of image
        for i in range(4):
            cropped_fiducial = MapCoords.crop(image, expected_positions_left[i], expected_positions_right[i])
            mask = MapCoords.preprocess_image(cropped_fiducial)
            edges = MapCoords.find_outline(mask)
            lines = self.draw_vertical_horizontal_lines(edges, original_image=cropped_fiducial, outputPath=MapCoords.output_dir+"/fiducialHoughLines"+str(i)+".jpg")
            corners = MapCoords.find_and_group_intersections(lines)
            corner_real_coords = []
            for corner in corners: corner_real_coords.append([corner[0] + expected_positions_left[i][0], corner[1] + expected_positions_left[i][1]])
            assert len(corners) == 4
            # MapCoords.plot_corners(cropped_fiducial, corners, MapCoords.output_dir + "/cornersfeducial" + str(i) + ".jpg")
            middle_corner = MapCoords.find_closest_corner(corner_real_coords, center)
            inner_corners.append(middle_corner)

        # print images
        MapCoords.plot_corners(image, inner_corners, MapCoords.output_dir+"/innerCorners.jpg")
        # find homography
        self.position_matrix = MapCoords.fill_position_matrix_tilted(image, inner_corners, cm_distance)
        return

    def get_real_coord(self, rx, ry):
        return self.position_matrix[rx][ry]

    def get_image(self):
        return self.image

    def reconstruct_image(self):
        cv2.imshow("test", self.image)
        scale = 10
        output = np.ones([self.cm_distance*scale+5, self.cm_distance*scale+5, 3])
        xSize, ySize, channels = self.image.shape
        for x in range(xSize):
            for y in range(ySize):
                if self.position_matrix[y][x] != -1:
                    temp = self.position_matrix[y][x]
                    outputX = int(temp[0] * scale)-1
                    outputY = int(temp[1] * scale)-1
                    output[outputX][outputY] = self.image[x][y]
                    if outputX == 300:
                        print(str(outputX) + " " + str(outputY) + " " + str(x) + " " + str(y) + " " + str(self.image[x][y]) + str(output[outputX][outputY]))

        cv2.imwrite('/Users/andylegrand/PycharmProjects/localization_image_testing/output/reim.jpg', output)

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

    @staticmethod
    def draw_vertical_horizontal_lines(edge_image, k=2, original_image = None, outputPath = None, **kwargs):
        rho, theta, thresh = 1, np.pi / 180, 50
        lines = cv2.HoughLines(edge_image, rho, theta, thresh)

        if original_image is not None:
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

            cv2.imwrite(outputPath, img)

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
        return segmented

    @staticmethod
    def find_and_group_intersections(lines):
        intersections = []
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
    def plot_corners(img, points, output_path):
        print(points)
        copy = img.copy()
        for point in points:
            copy = cv2.circle(copy, (point[0], point[1]), radius=5, color=(255, 0, 0), thickness=-1)
            cv2.putText(copy, "(x,y): " + str(point[0]) + ", " + str(point[1]),
                        (point[0], point[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.imwrite(output_path, copy)

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
                # print(str(pixel) + ' ' + str(position_matrix[pixel[0], pixel[1]]))
        return position_matrix

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
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:

            realCoord = self.ed.get_real_coord(x, y)

            # displaying the coordinates
            # on the Shell

            #print(realCoord[0],' ', realCoord[1],' ', x,' ', y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(self.img, str(round(realCoord[0],2)) + ',' +
                        str(round(realCoord[1],2)), (x, y), font,
                        1, (255, 0, 255), 2)

            cv2.imshow('image', self.img)

        # checking for right mouse clicks
        if event == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)

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
    # Read in the image
    img = cv2.imread('/Users/andylegrand/PycharmProjects/localization_image_testing/levitator_sample_images/object_images/undistorted2.jpg')
    #exLeft = [[400, 1000],[400, 2900],[2200, 830],[2200,3100]]
    #exRight = [[700, 1300],[700,3200],[2500,1130],[2500,3400]]

    exLeft = [[1000, 400], [2900,400], [830,2200], [3100,2200]]
    exRight = [[1300, 700], [3200, 700], [1130, 2500], [3400, 2500]]
    ed = MapCoords(img, exLeft, exRight, 60)
    visual_Test(ed)
    #ed.reconstruct_image()