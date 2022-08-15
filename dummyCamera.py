# for testing purposes. take pic method returns images from files instead of getting from camera
import cv2

full_res = (4056, 3040)

path1 = "/Users/andylegrand/PycharmProjects/objloc_ras_pi/object_images/object1.jpg"
path2 = "/Users/andylegrand/PycharmProjects/objloc_ras_pi/object_images/object2.jpg"

#path1 = "/Users/andylegrand/PycharmProjects/objloc_ras_pi/object_images/emptyLev1.jpg"
#path2 = "/Users/andylegrand/PycharmProjects/objloc_ras_pi/object_images/emptyLev2.jpg"

class DummyCamera:
    def __init__(self, cam_params, scale_factor):
        x = int(full_res[0] / scale_factor)
        y = int(full_res[1] / scale_factor)
        self.resolution = (x, y)

    def take_pic(self):
        im1 = cv2.imread(path1)
        im2 = cv2.imread(path2)

        assert im1 is not None and im2 is not None, "file paths not valid"

        im1 = cv2.resize(im1, self.resolution)
        im2 = cv2.resize(im2, self.resolution)

        return im1, im2