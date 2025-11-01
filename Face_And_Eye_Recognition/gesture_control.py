import cv2 as cv

# Read the image in grayscale
img = cv.imread(r"..\img\hand1.jpg", cv.IMREAD_GRAYSCALE)

# Apply thresholding to create a binary image
_, thresholded = cv.threshold(img, 70, 255, cv.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv.findContours(thresholded.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Convex Hull for each contour
convex_hulls = [cv.convexHull(contour) for contour in contours]

# Draw contours and convex hulls on the original image
original_with_contours = cv.drawContours(img.copy(), contours, -1, (0, 0, 0), 2)
original_with_convex_hulls = cv.drawContours(img.copy(), convex_hulls, -1, (0, 0, 0), 2)

# Display the images
cv.imshow("Original Image", img)
cv.imshow("Thresholded Image", thresholded)
cv.imshow("Contours", original_with_contours)
cv.imshow("Convex Hulls", original_with_convex_hulls)

# Wait for a key press and close windows
cv.waitKey(0)
cv.destroyAllWindows()
