import cv2 as cv
from cv2 import CascadeClassifier, VideoCapture
import numpy as np
import os
from typing import Tuple, List

def detect_faces_and_eyes(camera_index: int = 0, 
                         face_scale_factor: float = 1.1, 
                         face_min_neighbors: int = 7,
                         eye_scale_factor: float = 1.1, 
                         eye_min_neighbors: int = 7) -> None:
    """
    Detects faces and eyes in real-time using the webcam or a specified video source.
    
    Args:
        camera_index: Index of the camera to use (default is 0 for built-in webcam).
        face_scale_factor: Parameter specifying how much the image size is reduced at each image scale for face detection.
        face_min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it for face detection.
        eye_scale_factor: Parameter specifying how much the image size is reduced at each image scale for eye detection.
        eye_min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it for eye detection.
    
    Raises:
        FileNotFoundError: If the Haar cascade classifier files are not found.
        IOError: If the webcam cannot be opened.
    
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
    cap: VideoCapture = cv.VideoCapture(camera_index)

    if not cap.isOpened():
        raise IOError(f"Cannot open camera with index {camera_index}")

    try:
        while True:
            # Read a frame from the camera
            success: bool
            frame: np.ndarray
            success, frame = cap.read()

            if not success:
                print("Failed to grab frame")
                break

            # Convert the frame to grayscale for better performance
            gray_frame: np.ndarray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces: List[Tuple[int, int, int, int]] = face_cascade.detectMultiScale(
                gray_frame, 
                scaleFactor=face_scale_factor, 
                minNeighbors=face_min_neighbors,
                minSize=(30, 30),
                flags=cv.CASCADE_SCALE_IMAGE
            )

            # Detect eyes in the grayscale frame
            eyes: List[Tuple[int, int, int, int]] = eye_cascade.detectMultiScale(
                gray_frame, 
                scaleFactor=eye_scale_factor, 
                minNeighbors=eye_min_neighbors,
                minSize=(10, 10),
                flags=cv.CASCADE_SCALE_IMAGE
            )

            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw rectangles around detected eyes
            for (x, y, w, h) in eyes:
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

            # Display the resulting frame with detections
            cv.imshow("Face and Eye Detection", frame)

            # Check for the 'q' key to exit the loop
            key: int = cv.waitKey(1)
            if key == ord("q"):
                break

    finally:
        # Release the camera and close all OpenCV windows
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    # Call the main function with default parameters
    detect_faces_and_eyes()    