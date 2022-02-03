from PIL import Image
import os


class image2pdf:
    def __init__(self):
        self.validFormats = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")
        self.pictures = []
        self.files = os.listdir()
        self.convertPictures()
        input("Done ..... (Press Any Key To Exit)")

    def filter(self, item):
        return item.endswith(self.validFormats)

    def sortFiles(self):
        return sorted(self.files)

    def getPictures(self):
        pictures = list(filter(self.filter, self.sortFiles()))
        if self.isEmpty(pictures):
            print(" [Error] there are no pictrues in the directory ! ")
            raise Exception(" [Error] there are no pictrues in the directory !")
        print("pictures are : \n {}".format(pictures))
        return pictures

    def isEmpty(self, items):
        return True if len(items) == 0 else False

    def convertPictures(self):
        for picture in self.getPictures():
            self.pictures.append(Image.open(picture).convert("RGB"))
        self.save()

    def save(self):
        self.pictures[0].save(
            "result.pdf", save_all=True, append_images=self.pictures[1:]
        )


if __name__ == "__main__":
    image2pdf()
