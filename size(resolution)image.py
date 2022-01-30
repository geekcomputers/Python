def jpeg_res(filename):
    """ "This function prints the resolution of the jpeg image file passed into it"""

    # open image for reading in binary mode
    with open(filename, "rb") as img_file:

        # height of image (in 2 bytes) is at 164th position
        img_file.seek(163)

        # read the 2 bytes
        a = img_file.read(2)

        # calculate height
        height = (a[0] << 8) + a[1]

        # next 2 bytes is width
        a = img_file.read(2)

        # calculate width
        width = (a[0] << 8) + a[1]

    print("The resolution of the image is", width, "x", height)


jpeg_res("img1.jpg")
