System Overview:

Hardware Design:

Receiving video from raspberry pi:

In this system all the raspberry pi does is receive video from the hat and then send it to the pc. This is done via 
the hdmi port. Upon being turned on, the raspberry pi runs a script which fills the display with the camera output.
Instructions on how to do this can be found [here](https://webtechie.be/post/2021-12-20-raspberry-pi-as-hdmi-camera-for-atem-mini/)

One important thing to note is that the pi outputs at a 16:9 aspect ratio while the camera outputs at a 4:3 aspect 
ratio. While it is possible to fill the entire screen with the camera, this will cut off the top and bottom of the 
image (fig. 1). The best solution I found is to match the vertical resolutions and have black bars on either side 
(fig. 2). The camera output resolution can be changed by modifying the lib camera hello command in the startup script

Setting up the object localization software with a new hardware configuration (these steps only need to be performed once):

1. Run the findCameraIntrensics.py script. Place a 7x7 chessboard cutout beneath each camera, then press enter. The program will then attempt to find the chessboard corners for each camera. The program will then print if each camera was successful. Try to get at least 40 successful images per camera. More info about this calibration process can be found [here](https://www.geeksforgeeks.org/camera-calibration-with-python-opencv/),
2. Place the calibration plate inside the lev, then call the calibrate_focal_points method with an object localizer class. this will calculate the focal points of each camera with a given hardware setup

Running the object localization software:

After calibration the user should only need to interact with the object_localizer class. Each call to localize object 
inputs a raw frame from the pi and outputs the 3d coordinates of the object in millimeters. If no objects are found then None is returned

Overview of main algorithm used during localization:

Assume 1 object is in the levitator. Using opencvs find contour method we ae able to find the location of the projection 
of the object on the fiducial plane. We can then draw a vector from the cameras focal point to the location of the 
projection. The object is then located Where these two lines are closest to eachother. 

This looks like this:

Because there is no threshold value that will work for every background and object combination, different thresholds are 
tried until the object is detected. For each threshold value we recieve a set of contours from each camera. We then check 
every combination of contours and if the lines come close enough to eachother than an obejct is detected. To limit the 
amount of contours detected per thresholded image filters are applied. By default we filter by size and circularity. 
These filters are not meant to eliminate every contour which does not correspond to an object, but only those which are 
obviously wrong to help runtime.

