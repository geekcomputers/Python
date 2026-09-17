import cv2 as cv

# Read the image
img = cv.imread(r"..\img\hand1.jpg")

# Check if image loaded
if img is None:
    print("Error: Image not found")
    exit()

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply Gaussian Blur to remove noise
blur = cv.GaussianBlur(gray, (5, 5), 0)

# Apply Otsu Thresholding (automatic)
_, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Find contours
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Get largest contour (assumed as hand)
largest_contour = max(contours, key=cv.contourArea)

# Convex Hull
hull = cv.convexHull(largest_contour)

# Create copies for drawing
contour_img = img.copy()
hull_img = img.copy()

# Draw largest contour (Green)
cv.drawContours(contour_img, [largest_contour], -1, (0, 255, 0), 2)

# Draw convex hull (Blue)
cv.drawContours(hull_img, [hull], -1, (255, 0, 0), 2)

# Show images
cv.imshow("Original", img)
cv.imshow("Threshold", thresh)
cv.imshow("Largest Contour", contour_img)
cv.imshow("Convex Hull", hull_img)

# Wait & close
cv.waitKey(0)
cv.destroyAllWindows()
