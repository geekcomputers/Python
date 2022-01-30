import pyautogui  # pip install pyautogui
from PIL import Image, ImageGrab  # pip install pillow

# from numpy import asarray
import time


def hit(key):
    pyautogui.press(key)
    return


def isCollide(data):

    # for cactus
    for i in range(329, 425):
        for j in range(550, 650):
            if data[i, j] < 100:
                hit("up")
                return

    # Draw the rectangle for birds
    # for i in range(310, 425):
    #     for j in range(390, 550):
    #         if data[i, j] < 100:
    #             hit("down")
    #             return

    # return


if __name__ == "__main__":
    print("Hey.. Dino game about to start in 3 seconds")
    time.sleep(2)
    # hit('up')

    while True:
        image = ImageGrab.grab().convert("L")
        data = image.load()
        isCollide(data)

        # print(aarray(image))

        # Draw the rectangle for cactus
        # for i in range(315, 425):
        #     for j in range(550, 650):
        #         data[i, j] = 0

        # # # # # Draw the rectangle for birds
        # for i in range(310, 425):
        #     for j in range(390, 550):
        #         data[i, j] = 171

        # image.show()
        # break
