# Thumbnail Extractor

This Python function extracts a thumbnail frame from a video and saves it as an image file. It utilizes the OpenCV library to perform these operations. This README provides an overview of the function, its usage, and the required packages.

## Table of Contents
- [Function Description](#function-description)
- [Usage](#usage)
- [Required Packages](#required-packages)

## Function Description

The `extract_thumbnail` function takes two parameters:

- `video_path` (str): The path to the input video file.
- `frame_size` (tuple): A tuple containing the desired dimensions (width, height) for the thumbnail frame.

The function will raise an `Exception` if it fails to extract a frame from the video.

### Function Logic

1. The function opens the specified video file using OpenCV.
2. It seeks to the middle frame by calculating the middle frame index.
3. The frame is resized to the specified dimensions.
4. The resized frame is saved as an image file with a filename derived from the video's base name.

## Usage

Here's an example of how to use the function:

```python
from thumbnail_extractor import extract_thumbnail

# Extract a thumbnail from 'my_video.mp4' with dimensions (320, 240)
extract_thumbnail('my_video.mp4', (320, 240))
# Replace 'my_video.mp4' with the path to your own video file and (320, 240) with your desired thumbnail dimensions.

## Required Packages
```
To use this function, you need the following package:

- **OpenCV (cv2)**: You can install it using `pip`:

    ```shell
    pip install opencv-python
    ```

This function is useful for generating thumbnail images from videos. It simplifies the process of creating video thumbnails for various applications.


