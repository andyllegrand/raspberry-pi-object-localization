import cv2
import numpy as np

from imagePreprocessor import ImagePreprocessor
from objectLocalizer import ObjectLocalizer

import pickle

# load intrensic camera parameters. Used to undistort images from cameras
with open('cam1Params', 'rb') as f:
    cam1_params = pickle.load(f)
with open('cam2Params', 'rb') as f:
    cam2_params = pickle.load(f)

frame = cv2.imread("/Users/andylegrand/PycharmProjects/objloc_ras_pi/output/testframe.jpg")

image_preprocessor = ImagePreprocessor([cam1_params, cam2_params], (1000,1000))
im1, im2 = image_preprocessor.undistort_and_crop(frame)
cv2.imwrite("/Users/andylegrand/PycharmProjects/objloc_ras_pi/output/cam1/undistort.jpg", im1)
cv2.imwrite("/Users/andylegrand/PycharmProjects/objloc_ras_pi/output/cam2/undistort.jpg", im2)

test_im = cv2.imread("/Users/andylegrand/PycharmProjects/objloc_ras_pi/output/testframe.jpg")
blur = cv2.medianBlur(test_im, 5)
gs = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

filters = ObjectLocalizer.create_filters()

# apply blur
im1_blur = cv2.medianBlur(im1, 5)
im2_blur = cv2.medianBlur(im2, 5)

# convert to grayscale
im1_gray = cv2.cvtColor(im1_blur, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2_blur, cv2.COLOR_BGR2GRAY)

# Loop through threshold value until object is found
for counter in range(256):
    ret, thresh_im1 = cv2.threshold(im1_gray, counter, 255, 0)
    ret2, thresh_im2 = cv2.threshold(im2_gray, counter, 255, 0)

    contours_im1 = ObjectLocalizer.get_contours_and_apply_filters(thresh_im1, filters)
    contours_im2 = ObjectLocalizer.get_contours_and_apply_filters(thresh_im2, filters)

    if len(contours_im1) > 0:
        im1_copy = np.copy(im1)
        for c in range(len(contours_im1)):
            cv2.drawContours(im1_copy, [contours_im1[c]], 0, (0, 0, 255), 3)

        cv2.imwrite("/Users/andylegrand/PycharmProjects/objloc_ras_pi/thresh/cam1" + str(counter) +".jpg", im1_copy)


    if len(contours_im2) > 0:
        im2_copy = np.copy(im2)
        for c in range(len(contours_im2)):
            cv2.drawContours(im2_copy, [contours_im2[c]], 0, (0, 0, 255), 3)


        cv2.imwrite("/Users/andylegrand/PycharmProjects/objloc_ras_pi/thresh/cam2" + str(counter) + ".jpg", im2_copy)





