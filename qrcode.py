import qrcode
import cv2

qr = qrcode.QRCode(version=1, box_size=10, border=5)

data = input()
qr.add_data(data)
qr.make(fit=True)
img = qr.make_image(fill_color="blue", back_color="white")
path = data + ".png"
img.save(path)
cv2.imshow("QRCode", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
