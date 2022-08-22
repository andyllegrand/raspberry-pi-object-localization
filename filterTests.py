import cv2
import numpy as np
from objectLocalizer import ObjectLocalizer

test_im = cv2.imread("/Users/andylegrand/Downloads/IMG_1193.jpeg")
blur = cv2.medianBlur(test_im, 5)
gs = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

filters = ObjectLocalizer.create_filters()

for i in range(256):
    ret, thresh_im = cv2.threshold(gs, i, 255, cv2.THRESH_BINARY)
    im_copy = np.copy(test_im)
    contours = ObjectLocalizer.get_contours_and_apply_filters(thresh_im, filters)

    """
    cont_sizes = []
    for cont in contours:
        cont_sizes.append(cv2.contourArea(cont))
    """

    if len(contours) > 0:
        for c in range(len(contours)):
            cv2.drawContours(im_copy, [contours[c]], 0, (0, 0, 255), 3)

        cv2.imwrite("/Users/andylegrand/PycharmProjects/objloc_ras_pi/thresh/" + str(i) +".jpg", im_copy)


