# extract text from a img and its coordinates using the pytesseract module
import cv2
import pytesseract

# You need to add tesseract binary dependency to system variable for this to work

img = cv2.imread("img.png")
# We need to convert the img into RGB format
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

hI, wI, k = img.shape
print(pytesseract.image_to_string(img))
boxes = pytesseract.image_to_boxes(img)
for b in boxes.splitlines():
    b = b.split(" ")
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(img, (x, hI - y), (w, hI - h), (0, 0, 255), 0.2)

cv2.imshow("img", img)
cv2.waitKey(0)
