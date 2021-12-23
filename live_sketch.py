import cv2


def sketch(image):
    """
    Converts the input image to grayscale, applies a Gaussian blur and then uses Canny edge detection.

    :param img_gray: The input image converted to
    grayscale.
    :type img_gray: numpy array of 2 dimensions (a matrix) with 8-bit or floating-point elements.
    """
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    canny_edges = cv2.Canny(img_gray_blur, 10, 70)
    ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
    return mask


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Our Live Sketcher', sketch(frame))
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
