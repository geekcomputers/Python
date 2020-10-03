import pyqrcode
import PIL
from pyqrcode import QRCode
print("Enter text to convert")
s=input(": ")
print("Enter image name to save")
n=input(": ")
d=n+".png"
url=pyqrcode.create(s)
url.png(d, scale =6)