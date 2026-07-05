import qrcode


def generate_qr_code():
    url = input("Enter the URL: ").strip()

    if not url:
        print("No URL entered. Exiting.")
        return

    # before qrcode.png, add your desired path where it'll show the qrcode.
    # for eg: c:\\desktop\\myfile\\qrcode.png
    file_path = "qrcode.png"

    qr = qrcode.QRCode()
    qr.add_data(url)

    img = qr.make_image()
    img.save(file_path)

    print("QR code is generated!")


if __name__ == "__main__":
    generate_qr_code()