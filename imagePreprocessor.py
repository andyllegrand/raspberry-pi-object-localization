import cv2
import numpy as np

# points which define crop to isolate image from background
im_top_left = [240, 0]  # x, y
im_bottom_right = [1680, 1080]

im1_crop = np.array([[1550, 1005], [2900, 994], [1400, 2430], [3100, 2450]])
im2_crop = np.array([[837, 580], [3500, 630], [3200, 2850], [1100, 2800]])

camera_resolution = (4056, 3040)  # [columns, rows]


class ImagePreprocessor:
    """class for preprocessing images from camera setup described in readme"""
    def __init__(self, cam_params, desired_res):
        self.cam_params = cam_params
        self.desired_res = desired_res

    @staticmethod
    def get_cam_res():
        """returns camera resolution constant"""
        return camera_resolution

    @staticmethod
    def get_camera_images(img):
        """from hdmi input isolate camera images"""
        # crop out cli background
        cropped_im = img[im_top_left[1]:im_bottom_right[1], im_top_left[0]:im_bottom_right[0]]
        # cropped_im = img

        # divide 2 image and stretch along horizontal axis
        height, width, channels = cropped_im.shape
        middle = int(width / 2)
        im1 = cropped_im[:, :middle]
        im2 = cropped_im[:, middle:]

        # images compressed along x-axis. Stretch to undo this
        im1 = cv2.resize(im1, camera_resolution)
        im2 = cv2.resize(im2, camera_resolution)
        return im1, im2

    @staticmethod
    def four_point_crop(im, points):
        """crop image based off 4 corner points"""
        # actually returns rectangular crop, can modify in the future to return an actual 4 point crop

        # find smallest x, y and greatest x,y
        min_x = np.amin(points[0:4, 0])
        max_x = np.amax(points[0:4, 0])
        min_y = np.amin(points[0:4, 1])
        max_y = np.amax(points[0:4, 1])

        # create rectangular image
        cropped_im = im[min_y:max_y, min_x:max_x]

        return cropped_im

    def undistort(self, im):
        """given image input from setup described in README return an undistorted image from each camera"""
        # crop camera output
        im1, im2 = ImagePreprocessor.get_camera_images(im)

        # un-distort images
        if self.cam_params[0] is not None:
            camera_matrix1 = self.cam_params[0][0]
            distortion_coefficients1 = self.cam_params[0][1]
            scaled_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix1, distortion_coefficients1,
                                                                      camera_resolution, 1,
                                                                      camera_resolution)
            im1 = cv2.undistort(im1, camera_matrix1, distortion_coefficients1, None,
                                scaled_camera_matrix)

        if self.cam_params[1] is not None:
            camera_matrix2 = self.cam_params[1][0]
            distortion_coefficients2 = self.cam_params[1][1]
            scaled_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix2, distortion_coefficients2,
                                                                      camera_resolution, 1,
                                                                      camera_resolution)
            im2 = cv2.undistort(im2, camera_matrix2, distortion_coefficients2, None,
                                scaled_camera_matrix)

        return im1, im2

    def undistort_and_crop(self, im):
        """given image input from setup described in README return a cropped and undistorted image from each camera"""
        im1, im2 = self.undistort(im)

        # perform crops
        im1 = ImagePreprocessor.four_point_crop(im1, im1_crop)
        im2 = ImagePreprocessor.four_point_crop(im2, im2_crop)

        # resize to desired size
        im1 = cv2.resize(im1, self.desired_res)
        im2 = cv2.resize(im2, self.desired_res)
        return im1, im2

# test script. takes 2 undistorted pictures and saves them
if __name__ == '__main__':
    import pickle
    with open('cam1Params', 'rb') as f:
        cam1_params = pickle.load(f)
    with open('cam2Params', 'rb') as f:
        cam2_params = pickle.load(f)

    image_pre = ImagePreprocessor([cam1_params, cam2_params], (1000, 1000))

    # get first frame from video
    cap = cv2.VideoCapture('/Users/andylegrand/PycharmProjects/objloc_ras_pi/test.mp4')

    """
    # cycle 1 second
    for i in range(30): cap.read()

    ret, frame = cap.read()

    im1, im2 = image_pre.undistort_and_crop(frame)
    print(str(im1.shape))
    cv2.imwrite("/Users/andylegrand/PycharmProjects/objloc_ras_pi/output/object1.jpg", im1)
    cv2.imwrite("/Users/andylegrand/PycharmProjects/objloc_ras_pi/output/object2.jpg", im2)
    print("done")
    """

    crop = image_pre.get_camera_images(cv2.imread("/Users/andylegrand/PycharmProjects/objloc_ras_pi/test_images/testframe0.jpg"))
    cv2.imshow("test", crop)

