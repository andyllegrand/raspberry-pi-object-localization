import cv2
import numpy as np

class objectDetector:
    def __init__(self):
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 200
        params.thresholdStep = 5
        params.minRepeatability = 10

        params.filterByColor = True
        params.blobColor = 0

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 50

        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.87

        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.01

        params.minDistBetweenBlobs = 300

        self.detector = cv2.SimpleBlobDetector_create(params)

    def detectObject(self, emptyLev, Object):
        emptyLevGreyScale = cv2.cvtColor(emptyLev, cv2.COLOR_BGR2GRAY)
        ObjectGreyScale = cv2.cvtColor(Object, cv2.COLOR_BGR2GRAY)

        emptyLevGreyScaleBlur = cv2.GaussianBlur(emptyLevGreyScale, (0, 0), cv2.BORDER_DEFAULT)
        ObjectGreyScaleBlur = cv2.GaussianBlur(ObjectGreyScale, (0, 0), cv2.BORDER_DEFAULT)

        sub = 255 - cv2.subtract(ObjectGreyScaleBlur, emptyLevGreyScaleBlur)
        # cv2.imshow("sub", sub)
        # cv2.waitKey(0)
        keypoints = self.detector.detect(sub)
        print(keypoints)
        im_with_keypoints = cv2.drawKeypoints(sub, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # Show keypoints
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.waitKey(0)
        return keypoints

if __name__ == '__main__':
    ed = objectDetector()

    empty = cv2.imread('/Users/andylegrand/PycharmProjects/localization_image_testing/levitator_sample_images/object_images/cam1_img_2.jpg')
    object = cv2.imread('/Users/andylegrand/PycharmProjects/localization_image_testing/levitator_sample_images/object_images/cam1_img_4.jpg')

    points = ed.detectObject(empty, object)