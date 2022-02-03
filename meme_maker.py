import sys

from PIL import ImageDraw, ImageFont, Image


def input_par():
    print("Enter the text to insert in image: ")
    text = str(input())
    print("Enter the desired size of the text: ")
    size = int(input())
    print("Enter the color for the text(r, g, b): ")
    color_value = [int(i) for i in input().split(" ")]
    return text, size, color_value
    pass


def main():
    path_to_image = sys.argv[1]
    image_file = Image.open(path_to_image + ".jpg")
    image_file = image_file.convert("RGBA")
    pixdata = image_file.load()

    print(image_file.size)
    text, size, color_value = input_par()

    # Font path is given as -->( " Path  to  your  desired  font " )
    font = ImageFont.truetype("C:\\Windows\\Fonts\\Arial.ttf", size=size)

    # If the color of the text is not equal to white,then change the background to be white
    if (color_value[0] and color_value[1] and color_value[2]) != 255:
        for y in range(100):
            for x in range(100):
                pixdata[x, y] = (255, 255, 255, 255)
    # If the text color is white then the background is said to be black
    else:
        for y in range(100):
            for x in range(100):
                pixdata[x, y] = (0, 0, 0, 255)
    image_file.show()

    # Drawing text on the picture
    draw = ImageDraw.Draw(image_file)
    draw.text(
        (0, 2300), text, (color_value[0], color_value[1], color_value[2]), font=font
    )
    draw = ImageDraw.Draw(image_file)

    print("Enter the file name: ")
    file_name = str(input())
    image_file.save(file_name + ".jpg")
    pass


if __name__ == "__main__":
    main()
