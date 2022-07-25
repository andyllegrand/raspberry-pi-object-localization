from cameraClass import Camera
import cv2
import os

counter = 0
camera = Camera()

parent_dir = "/home/pi/piObjLoc/"
#dir_name = "chessboard_images"
dir_name = "object_images"
full_path = parent_dir+dir_name

# make directory. Delete and recreate if already exists
try:
    os.mkdir(full_path)
except OSError as error:
    print("directory already exists")

os.chdir(full_path)

while True:
    inp = input("any key to take picture s to stop")
    if inp == "s":
        break
    im1, im2 = camera.take_pic()
    cv2.imwrite("cam1_img_"+str(counter)+".jpg", im1)
    cv2.imwrite("cam2_img_"+str(counter)+".jpg", im2)
    print(str(counter) + " images captured \n")
    counter+=1
