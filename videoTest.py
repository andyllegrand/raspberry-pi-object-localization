from picamera2 import Picamera2, MappedArray
import cv2
import time

backSub = cv2.createBackgroundSubtractorMOG2()

# draw circle around object and write estimated coordinates
def draw_boxes(request):
    return

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (160, 120)})
picam2.configure(config)
(w0, h0) = picam2.stream_configuration("main")["size"]
faces = []
picam2.post_callback = draw_boxes
picam2.start(show_preview=True)
while True:
    input("ready")
    array = picam2.capture_array("main") # blue and red swapped but shouldn't matter
    print("done")

