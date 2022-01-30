# importing Required Modules
import qrcode

# QR Code Generator
query = input("Enter Content: ")  # Enter Content
code = qrcode.make(str(query))  # Making the QR code
code.save("qrcode.png")  # Saving the QR code file
