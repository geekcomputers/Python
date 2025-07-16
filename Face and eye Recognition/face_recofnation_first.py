import cv2 as cv
from cv2 import CascadeClassifier, VideoCapture
import numpy as np
import os

def detect_faces_and_eyes() -> None:
    """
    Detects faces and eyes in real-time using the webcam.
    Press 'q' to exit the program.
    """
    # Get the directory where OpenCV's data files are located
    cv2_data_dir = os.path.join(os.path.dirname(cv.__file__), 'data')
    
    # Construct absolute paths to the Haar cascade files
    face_cascade_path = os.path.join(cv2_data_dir, 'haarcascade_frontalface_default.xml')
    eye_cascade_path = os.path.join(cv2_data_dir, 'haarcascade_eye.xml')

    # Load the pre-trained classifiers for face and eye detection
    face_cascade: CascadeClassifier = cv.CascadeClassifier(face_cascade_path)
    eye_cascade: CascadeClassifier = cv.CascadeClassifier(eye_cascade_path)

    # Validate classifier loading
    if face_cascade.empty() or eye_cascade.empty():
        raise FileNotFoundError(f"Unable to load Haar cascade classifiers from:\n"
                               f"Face: {face_cascade_path}\n"
                               f"Eye: {eye_cascade_path}\n\n"
                               f"Please verify the files exist in your OpenCV installation.")

    # Open the webcam
    cap: VideoCapture = cv.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    try:
        while True:
            # Read a frame from the webcam
            flag: bool
            img: np.ndarray
            flag, img = cap.read()

            if not flag:
                print("Failed to grab frame")
                break

            # Convert the frame to grayscale for better performance
            gray: np.ndarray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces: np.ndarray = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=7
            )

            # Detect eyes in the frame
            eyes: np.ndarray = eye_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=7
            )

            # Draw rectangles around faces and eyes
            for (x, y, w, h) in faces:
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

            for (a, b, c, d) in eyes:
                cv.rectangle(img, (a, b), (a + c, b + d), (255, 0, 0), 1)

            # Display the resulting frame
            cv.imshow("Face and Eye Detection", img)

            # Check for the 'q' key to exit the program
            key: int = cv.waitKey(1)
            if key == ord("q"):
                break

    finally:
        # Release the webcam and close all windows
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    # Call the main function
    detect_faces_and_eyes()