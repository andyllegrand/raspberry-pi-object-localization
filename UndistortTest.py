import cv2
import numpy as np
from imagePreprocessor import ImagePreprocessor
import pickle

# load intrensic camera parameters. Used to undistort images from cameras
with open('cam1Params', 'rb') as f:
    cam1_params = pickle.load(f)
with open('cam2Params', 'rb') as f:
    cam2_params = pickle.load(f)

# get first frame from video
cap = cv2.VideoCapture('/Users/andylegrand/PycharmProjects/objloc_ras_pi/test.mp4')
ret, frame = cap.read()

image_preprocessor = ImagePreprocessor([cam1_params, cam2_params], None, None, None)
im1, im2 = image_preprocessor.undistort(frame)
cv2.imwrite("/Users/andylegrand/PycharmProjects/objloc_ras_pi/output/cam1/undistort.jpg", im1)
cv2.imwrite("/Users/andylegrand/PycharmProjects/objloc_ras_pi/output/cam2/undistort.jpg", im2)

