# Importing Required Modules
import cv2

# QR Code Decoder

filename = input()
image = cv2.imread(filename)  # Enter name of the image
detector = cv2.QRCodeDetector()
data, vertices_array, binary_qrcode = detector.detectAndDecode(image)
if vertices_array is not None:
    print(data)
