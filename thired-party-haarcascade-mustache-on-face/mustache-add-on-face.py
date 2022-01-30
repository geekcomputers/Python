import cv2

from utils import image_resize

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("face.xml")

nose_cascade = cv2.CascadeClassifier("Nose.xml")

mustache = cv2.imread("image/mustache.png", -1)

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    for (x, y, w, h) in faces:
        roi_gray = gray[y : y + h, x : x + h]  # rec
        roi_color = frame[y : y + h, x : x + h]

        nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
        for (nx, ny, nw, nh) in nose:

            roi_nose = roi_gray[ny : ny + nh, nx : nx + nw]
            mustache2 = image_resize(mustache.copy(), width=nw)

            mw, mh, mc = mustache2.shape
            for i in range(0, mw):
                for j in range(0, mh):

                    if mustache2[i, j][3] != 0:  # alpha 0
                        roi_color[ny + int(nh / 2.0) + i, nx + j] = mustache2[i, j]

    # Display the resulting frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    cv2.imshow("frame", frame)
    if cv2.waitKey(20) & 0xFF == ord("x"):
        break

cap.release()
cv2.destroyAllWindows()
