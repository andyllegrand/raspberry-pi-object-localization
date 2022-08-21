import cv2
import numpy as np

# while this kind of works it does not seem very robust. Doing a crop might be a good idea

class objectDetector:
    def __init__(self):
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 55
        params.maxThreshold = 255
        params.thresholdStep = 10
        params.minRepeatability = 2

        params.filterByColor = False
        params.blobColor = 0

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 10

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.8

        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.87

        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.01

        self.detector = cv2.SimpleBlobDetector_create(params)

    # TODO does not work with white background
    def detectObject(self, emptyLev, lev_with_object, path):
        emptyLevGreyScale = cv2.cvtColor(emptyLev, cv2.COLOR_BGR2GRAY)
        ObjectGreyScale = cv2.cvtColor(lev_with_object, cv2.COLOR_BGR2GRAY)

        emptyLevGreyScaleBlur = cv2.GaussianBlur(emptyLevGreyScale, (0, 0), cv2.BORDER_DEFAULT)
        ObjectGreyScaleBlur = cv2.GaussianBlur(ObjectGreyScale, (0, 0), cv2.BORDER_DEFAULT)

        sub = 255 - cv2.subtract(ObjectGreyScaleBlur, emptyLevGreyScaleBlur)
        cv2.imwrite(path+"/sub.jpg", sub)
        # cv2.imshow("sub", sub)
        # cv2.waitKey(0)
        keypoints = self.detector.detect(sub)

        # Show keypoints
        print(keypoints)
        im_with_keypoints = cv2.drawKeypoints(lev_with_object, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(path + "/detect.jpg", im_with_keypoints)

        assert len(keypoints) == 1

        ret = [int(keypoints[0].pt[0]), int(keypoints[0].pt[1])]
        return ret

if __name__ == '__main__':
    ed = objectDetector()

    empty = cv2.imread('/home/pi/piObjLocSync/object_images/cam1/img_0.jpg')
    object = cv2.imread('/home/pi/piObjLocSync/object_images/cam1/img_1.jpg')

    points = ed.detectObject(empty, object)