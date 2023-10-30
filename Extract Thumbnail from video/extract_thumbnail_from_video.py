import cv2
import os

def extract_thumbnail(video_path, frame_size):
    """
    Extracts a thumbnail frame from a video and saves it as an image file.

    Args:
        video_path (str): The path to the input video file.
        frame_size (tuple): A tuple containing the desired dimensions (width, height) for the thumbnail frame.

    Raises:
        Exception: If the function fails to extract a frame from the video.

    The function opens the specified video file, seeks to the middle frame,
    resizes the frame to the specified dimensions, and saves it as an image
    file with a filename derived from the video's base name.

    Example:
        extract_thumbnail('my_video.mp4', (320, 240))
    
    Required Packages:
        cv2 (pip install cv2)
    
    This function is useful for generating thumbnail images from videos.
    """
    video_capture = cv2.VideoCapture(video_path)  # Open the video file for reading
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames in the video
    middle_frame_index = total_frames // 2  # Calculate the index of the middle frame
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)  # Seek to the middle frame
    success, frame = video_capture.read()  # Read the middle frame
    video_capture.release()  # Release the video capture object

    if success:
        frame = cv2.resize(frame, frame_size)  # Resize the frame to the specified dimensions
        thumbnail_filename = f"{os.path.basename(video_path)}_thumbnail.jpg"  # Create a filename for the thumbnail
        cv2.imwrite(thumbnail_filename, frame)  # Save the thumbnail frame as an image
    else:
        raise Exception("Could not extract frame")  # Raise an exception if frame extraction fails
