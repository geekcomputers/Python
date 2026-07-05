import qrcode

url = input("Enter the URL: ").strip()
# before qrcode.png add you desired path where it'll show you qrcode.
# for eg : c:\\desktop\\myfile\\qrcode.png
file_path = "qrcode.png"

qr = qrcode.QRCode()
qr.add_data(url)

img = qr.make_image()
img.save(file_path)

print("QR code is generated!")