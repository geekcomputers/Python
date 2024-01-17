import cv2 as cv


def detect_faces_and_eyes():
    """
    Detects faces and eyes in real-time using the webcam.

    Press 'q' to exit the program.
    """
    # Load the pre-trained classifiers for face and eye detection
    face_cascade = cv.CascadeClassifier(r"..\libs\haarcascade_frontalface_default.xml")
    eye_cascade = cv.CascadeClassifier(r"..\libs\haarcascade_eye.xml")

    # Open the webcam
    cap = cv.VideoCapture(0)

    while cap.isOpened():
        # Read a frame from the webcam
        flag, img = cap.read()

        # Convert the frame to grayscale for better performance
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

        # Detect eyes in the frame
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

        # Draw rectangles around faces and eyes
        for x, y, w, h in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

        for a, b, c, d in eyes:
            cv.rectangle(img, (a, b), (a + c, b + d), (255, 0, 0), 1)

        # Display the resulting frame
        cv.imshow("Face and Eye Detection", img)

        # Check for the 'q' key to exit the program
        key = cv.waitKey(1)
        if key == ord("q"):
            break

    # Release the webcam and close all windows
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    # Call the main function
    detect_faces_and_eyes()