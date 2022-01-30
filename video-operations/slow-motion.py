# author : Avee Chakraborty
# Department of software engineering, Diu
# Bangladesh

import cv2

capture = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output = cv2.VideoWriter("slowmotion.mp4", fourcc, 5, (640, 480))

while True:
    ret, frame = capture.read()
    output.write(frame)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("x"):
        break

capture.release()
output.release()
cv2.destroyAllWindows()
