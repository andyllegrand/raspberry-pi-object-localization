import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
# cap = cv2.VideoCapture("/Users/andylegrand/PycharmProjects/objloc_ras_pi/test.mp4") # this is the magic!
cap = cv2.VideoCapture(0)
counter = 0

# Check if camera opened successfully
if (cap.isOpened() == False):
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  #for i in range(15):
  ret, frame = cap.read()

  if ret == True:
    """
    height, width, channels = frame.shape
    middle = int(width / 2)
    im1 = frame[:, :middle]
    frame = cv2.resize(im1, (width, height))
    """

    #cv2.imwrite("/Users/andylegrand/PycharmProjects/objloc_ras_pi/test_images/testframe" + str(counter) + ".jpg", frame)
    cv2.imshow("frame", frame)
    counter+=1
    cv2.waitKey(5)

  # Break the loop
  else:
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()