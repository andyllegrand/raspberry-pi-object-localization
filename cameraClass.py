from picamera2 import Picamera2
import cv2

full_res = (4056, 3040)

class Camera:
    def __init__(self, cam_params, scale_factor):
        x = int(full_res[0] / scale_factor)
        y = int(full_res[1] / scale_factor)
        self.resolution = (x, y)
        self.cam_params = cam_params
        self.picam2 = Picamera2()
        capture_config = self.picam2.create_still_configuration({"size": full_res})
        self.picam2.configure(capture_config)

    # image is captured in full res, undistorted, then resized.
    def take_pic(self):
        self.picam2.start()
        np_array = self.picam2.capture_array()
        self.picam2.stop()

        # change from rgb to bgr
        open_cv_image = np_array[:, :, ::-1].copy()

        # crop image and stretch along horizontal axis
        height, width, channels = open_cv_image.shape
        middle = int(width/2)
        im1 = open_cv_image[:,:middle]
        im2 = open_cv_image[:,middle:]

        # need to split image before resizing due to bug in opencv https://stackoverflow.com/questions/31996367/opencv-resize-fails-on-large-image-with-error-215-ssize-area-0-in-funct
        im1 = cv2.resize(im1, (width, height))
        im2 = cv2.resize(im2, (width, height))

        if self.cam_params[0] is not None:
            # undistort using intrensic params
            camera_matrix1 = self.cam_params[0][0]
            distortion_coefficients1 = self.cam_params[0][1]
            scaled_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix1, distortion_coefficients1, (width, height), 1,
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

        im1 = cv2.resize(im1, self.resolution)
        im2 = cv2.resize(im2, self.resolution)
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
    camera = Camera([cam1_params, cam2_params], 1)

    im1, im2 = camera.take_pic()
    print(str(im1.shape))
    cv2.imwrite("/home/pi/piObjLocSync/output/object1.jpg", im1)
    cv2.imwrite("/home/pi/piObjLocSync/output/object2.jpg", im2)
    print("done")