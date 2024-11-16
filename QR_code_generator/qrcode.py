import pyqrcode, png
# from pyqrcode import QRCode 
# no need to import same library again and again

# Creating QR code after given text "input"
url = pyqrcode.create(input("Enter text to convert: "))
# Saving QR code as a png file
url.show()
# Name of QR code png file "input"
url.png(input("Enter image name to save: ") + ".png", scale=6)
