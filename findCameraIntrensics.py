import os
import numpy as np
import cv2 as cv
import pickle
from imagePreprocessor import ImagePreprocessor

camera_input = 0
cap = cv.VideoCapture(camera_input)

cam1_output_dir = "/Users/andylegrand/PycharmProjects/objloc_ras_pi/chessboard_images/cam1"
cam2_output_dir = "/Users/andylegrand/PycharmProjects/objloc_ras_pi/chessboard_images/cam2"

cam1_num = 0
cam2_num = 0

def add_image(img, objpoints, imgpoints):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,7), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print("works")
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,7), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
    else:
        print("removed")
    return ret

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(7,7,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpointsCam1 = [] # 3d point in real world space
imgpointsCam1 = [] # 2d points in image plane.

# Arrays to store object points and image points from all the images.
objpointsCam2 = [] # 3d point in real world space
imgpointsCam2 = [] # 2d points in image plane.

while True:
    inp = input("enter to take photo s to stop")
    if inp == "s":
        break

    img = cap.read()
    im1, im2 = ImagePreprocessor.get_camera_images(img)

    if add_image(im1, objpointsCam1, imgpointsCam1):
        cv.imwrite(cam1_output_dir+"img"+str(cam1_num)+".jpg", im1)
        cam1_num += 1

    if add_image(im2, objpointsCam2, imgpointsCam2):
        cv.imwrite(cam1_output_dir+"img"+str(cam2_num)+".jpg", im2)
        cam2_num += 1


ret1, mtx1, dist1, rvecs1, tvecs1 = cv.calibrateCamera\
    (objpointsCam1, imgpointsCam1, ImagePreprocessor.get_cam_res(), None, None)
print(mtx1)
print(dist1)
with open('cam1Params', 'wb') as f:
    pickle.dump([mtx1, dist1], f)

ret2, mtx2, dist2, rvecs2, tvecs2 = cv.calibrateCamera\
    (objpointsCam2, imgpointsCam2, ImagePreprocessor.get_cam_res(), None, None)
print(mtx2)
print(dist2)
with open('cam2Params', 'wb') as f:
    pickle.dump([mtx2, dist2], f)