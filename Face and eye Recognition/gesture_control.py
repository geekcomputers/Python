
import cv2 as cv
import numpy as np


def process_hand_image(image_path: str) -> None:
    """
    Processes an image to detect hand contours and convex hulls.
    
    Args:
        image_path: Path to the input image file.
    
    Raises:
        FileNotFoundError: If the specified image file cannot be found.
        cv.error: If there is an error during image processing.
    """
    # Read the image in grayscale
    img: np.ndarray = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    # Validate image loading
    if img is None:
        raise FileNotFoundError(f"Could not read image from path: {image_path}")
    
    # Apply thresholding to create a binary image
    thresholded: np.ndarray
    _, thresholded = cv.threshold(img, 70, 255, cv.THRESH_BINARY)
    
    # Find contours in the binary image
    contours: list[np.ndarray]
    hierarchy: np.ndarray
    contours, hierarchy = cv.findContours(thresholded.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Compute convex hull for each contour
    convex_hulls: list[np.ndarray] = [cv.convexHull(contour) for contour in contours]
    
    # Convert grayscale image to BGR for colored drawing
    color_img: np.ndarray = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    
    # Draw contours and convex hulls on the colored image
    contours_image: np.ndarray = color_img.copy()
    cv.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
    
    convex_hulls_image: np.ndarray = color_img.copy()
    cv.drawContours(convex_hulls_image, convex_hulls, -1, (0, 0, 255), 2)
    
    # Display the images
    cv.imshow("Original Image", img)
    cv.imshow("Thresholded Image", thresholded)
    cv.imshow("Contours", contours_image)
    cv.imshow("Convex Hulls", convex_hulls_image)
    
    # Wait for a key press and then close all windows
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    process_hand_image(r"..\img\hand1.jpg")    