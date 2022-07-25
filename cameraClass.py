from picamera2 import Picamera2
import cv2
import os,sys

class Camera:
    def __init__(self, distortion_params = None):
        self.picam2 = Picamera2()
        capture_config = self.picam2.create_still_configuration()
        self.picam2.configure(capture_config)

    def take_pic(self):
        self.picam2.start()
        np_array = self.picam2.capture_array()

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

        self.picam2.stop()
        return im1, im2
