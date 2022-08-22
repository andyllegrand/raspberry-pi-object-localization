import cv2
import numpy as np


class ImagePreprocessor:
    def __init__(self, cam_params, desired_res, im1crop, im2crop):
        self.cam_params = cam_params
        self.desired_res = desired_res
        self.im1crop = im1crop
        self.im2crop = im2crop

    # actually returns rectangular crop, can modify in the future to return an actual 4 point crop
    @staticmethod
    def four_point_crop(im, points):
        # find smallest x, y and greatest x,y
        min_x = np.amin(points[:0])
        max_x = np.amax(points[:0])
        min_y = np.amin[points[:1]]
        max_y = np.amax[points[:1]]

        # create rectangular image
        cropped_im = im[min_y:max_y, min_x:max_x]

        return cropped_im

    def undistort(self, im):
        # crop image and stretch along horizontal axis
        height, width, channels = im.shape
        middle = int(width / 2)
        im1 = im[:, :middle]
        im2 = im[:, middle:]

        # images compressed along x-axis. Stretch to undo this
        im1 = cv2.resize(im1, (width, height))
        im2 = cv2.resize(im2, (width, height))

        # un-distort images
        if self.cam_params[0] is not None:
            camera_matrix1 = self.cam_params[0][0]
            distortion_coefficients1 = self.cam_params[0][1]
            scaled_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix1, distortion_coefficients1,
                                                                      (width, height), 1,
                                                                      (width, height))
            im1 = cv2.undistort(im1, camera_matrix1, distortion_coefficients1, None,
                                scaled_camera_matrix)

        if self.cam_params[1] is not None:
            camera_matrix2 = self.cam_params[1][0]
            distortion_coefficients2 = self.cam_params[1][1]
            scaled_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix2, distortion_coefficients2,
                                                                      (width, height), 1,
                                                                      (width, height))
            im2 = cv2.undistort(im2, camera_matrix2, distortion_coefficients2, None,
                                scaled_camera_matrix)

        return im1, im2

    # gets image, undistorts, crops, and resizes. Returns 2 processed images
    def undistort_and_crop(self, im):
        im1, im2 = self.undistort(im)

        # perform crops
        im1 = ImagePreprocessor.four_point_crop(im1, self.im1crop)
        im2 = ImagePreprocessor.four_point_crop(im2, self.im2crop)

        # resize to desired size
        im1 = cv2.resize(im1, self.desired_res)
        im2 = cv2.resize(im2, self.desired_res)
        return im1, im2

# test script. takes 2 undistorted pictires and saves them
if __name__ == '__main__':
    import pickle
    with open('cam1Params', 'rb') as f:
        cam1_params = pickle.load(f)
    with open('cam2Params', 'rb') as f:
        cam2_params = pickle.load(f)

    #cam2_params = cam1_params

    print("starting cameras...")
    camera = ImagePreprocessor([cam1_params, cam2_params], 1)

    im1, im2 = camera.take_pic()
    print(str(im1.shape))
    cv2.imwrite("/home/pi/piObjLocSync/output/object1.jpg", im1)
    cv2.imwrite("/home/pi/piObjLocSync/output/object2.jpg", im2)
    print("done")

