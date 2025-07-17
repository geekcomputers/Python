import cv2
import os
from typing import Tuple

def extract_thumbnail(video_path: str, frame_size: Tuple[int, int]) -> None:
    """
    Extracts a thumbnail frame from a video and saves it as an image file.

    Args:
        video_path: The path to the input video file.
        frame_size: A tuple containing the desired dimensions (width, height) for the thumbnail frame.

    Raises:
        Exception: If the function fails to extract a frame from the video.

    The function opens the specified video file, seeks to the middle frame,
    resizes the frame to the specified dimensions, and saves it as an image
    file with a filename derived from the video's base name.

    Example:
        extract_thumbnail('my_video.mp4', (320, 240))
    
    Required Packages:
        OpenCV (pip install opencv-python)
    
    This function is useful for generating thumbnail images from videos.
    """
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    try:
        # Get the total number of frames
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate the middle frame index
        middle_frame_index = total_frames // 2
        
        # Seek to the middle frame
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
        
        # Read the frame
        success, frame = video_capture.read()
        
        if not success:
            raise Exception("Failed to read frame from video")
        
        # Resize the frame to the specified dimensions
        resized_frame = cv2.resize(frame, frame_size)
        
        # Generate the thumbnail filename
        base_name = os.path.basename(video_path)
        file_name, _ = os.path.splitext(base_name)
        thumbnail_filename = f"{file_name}_thumbnail.jpg"
        
        # Save the thumbnail image
        cv2.imwrite(thumbnail_filename, resized_frame)
        
        print(f"Thumbnail saved as: {thumbnail_filename}")
        
    finally:
        # Release the video capture object
        video_capture.release()    