from cameraClass import Camera
from MapCoords import MapCoords
import cv2
import os
import pickle

with open('cam1Params', 'rb') as f:
    cam1_params = pickle.load(f)
with open('cam2Params', 'rb') as f:
    cam2_params = pickle.load(f)

cam2_params = cam1_params # remove this!!!
camera = Camera([cam1_params,cam2_params])

counter = 0

full_path = "/home/pi/piObjLocSync/object_images/"

os.chdir(full_path)

while True:
    inp = input("any key to take picture s to stop")
    if inp == "s":
        break
    im1, im2 = camera.take_pic()
    cv2.imwrite("cam1/img_"+str(counter)+".jpg", im1)
    cv2.imwrite("cam2/img_"+str(counter)+".jpg", im2)
    print(str(counter) + " images captured \n")
    counter+=1
