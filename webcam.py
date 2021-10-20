# Requirements:
#     pip install numpy
#     sudo apt-get install python-openCV
# Program: 
#     opens your webcam, and records.  

import cv2

cap = cv2.VideoCapture(0)

# Obtain resolutions, convert resolutions from float to integer
frames_width = int(cap.get(3))
frames_height = int(cap.get(4))

# Specify the video codec
# FourCC is plateform dependent, however MJPG is a safe choice.
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

# Create video writer object. Save file to recording.avi
out = cv2.VideoWriter('recording.avi', fourcc, 20.0, (frames_width, frames_height))

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:

        # Write frame to recording.avi
        out.write(frame)

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture and video writer
cap.release()
out.release()
cv2.destroyAllWindows()

