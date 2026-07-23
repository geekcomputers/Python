# Requirements:
#     pip install numpy
#     pip install opencv-python
# Program:
#     Opens your webcam and records.

# Improve this program and make it suitable for general module like use in another programs
import cv2
from colorama import Fore

cap = cv2.VideoCapture(0)

# Obtain resolutions, convert resolutions from float to integer
frames_width = int(cap.get(3))
frames_height = int(cap.get(4))

# Specify the video codec
# FourCC is platform dependent; however, MJPG is a safe choice.
fourcc = cv2.VideoWriter_fourcc(*"MJPG")

# exception Handling for captured video
try:
    # 60 FPS video capture
    # Create video writer object. Save file to recording.avi
    out = cv2.VideoWriter("recording.avi", fourcc, 60.0, (frames_width, frames_height))
except(Exception) as e:
    print(Fore.RED, e, Fore.RESET)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:
        # Write frame to recording.avi
        out.write(frame)
        
        # color video output
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        # Display the resulting frame
        cv2.imshow("frame", gray)
        if cv2.waitKey(1) & 0xFF == ord(" "):
            break

# When everything is done, release the capture and video writer
cap.release()
out.release()
cv2.destroyAllWindows()

