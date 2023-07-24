from PIL import Image
import os


class image2pdf:
    def __init__(self):
        self.validFormats = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")
        self.pictures = []
        
        self.directory = ""
        self.isMergePDF = True 


    def getUserDir(self):
        """ Allow user to choose image directory """

        msg = "\n1. Current directory\n2. Custom directory\nEnter a number: "
        user_option = int(input(msg))

        # Restrict input to either (1 or 2)
        while user_option <= 0 or user_option >= 3:
            user_option = int(input(f"\n*Invalid input*\n{msg}"))

        self.directory = os.getcwd() if user_option == 1 else input("\nEnter custom directory: ")
        
    def filter(self, item):
        return item.endswith(self.validFormats)

    def sortFiles(self):
        return sorted(os.listdir(self.directory))

    def getPictures(self):
        pictures = list(filter(self.filter, self.sortFiles()))

        if not pictures:
            print(f" [Error] there are no pictures in the directory: {self.directory} ")
            return False
        
        print(f"Found picture(s) :")
        return pictures

    def selectPictures(self, pictures):
        """ Allow user to manually pick each picture or merge all """

        listedPictures = {}
        for index, pic in enumerate(pictures):
            listedPictures[index+1] = pic
            print(f"{index+1}: {pic}")
        
        userInput = input("\n Enter the number(s) - (comma seperated/no spaces) or (A or a) to merge All \nChoice: ").strip().lower()
        
        if userInput != "a":
            # Convert user input (number) into corresponding (image title)
            pictures = (
                listedPictures.get(int(number)) for number in userInput.split(',')
            )

            self.isMergePDF = False

        return pictures

    
    def convertPictures(self):
        """
            Convert pictures according the following:
            * If pictures = 0 -> Skip 
            * If pictures = 1 -> use all 
            * Else            -> allow user to pick pictures

            Then determine to merge all or one pdf
        """

        pictures = self.getPictures()
        totalPictures = len(pictures) if pictures else 0
        
        if totalPictures == 0:
            return
        
        elif totalPictures >= 2:
            pictures = self.selectPictures(pictures)
        
        if self.isMergePDF:
            # All pics in one pdf. 
            for picture in pictures:
                self.pictures.append(Image.open(f"{self.directory}\\{picture}").convert("RGB"))
            self.save()

        else:
            # Each pic in seperate pdf. 
            for picture in pictures:
                self.save(Image.open(f"{self.directory}\\{picture}").convert("RGB"), picture, False)

        # Reset to default value for next run
        self.isMergePDF = True
        self.pictures = []
        print(f"\n{'#'*30}")
        print("            Done! ")
        print(f"{'#'*30}\n")

    def save(self, image=None, title="All-PDFs", isMergeAll=True):
        # Save all to one pdf or each in seperate file

        if isMergeAll:
            self.pictures[0].save(
                f"{self.directory}\\{title}.pdf", 
                save_all=True, 
                append_images=self.pictures[1:]
            )
        
        else:
            image.save(f"{self.directory}\\{title}.pdf")


if __name__ == "__main__":
    
    # Get user directory only once
    process = image2pdf()
    process.getUserDir()
    process.convertPictures()

    # Allow user to rerun any process
    while True:
        user = input("Press (R or r) to Run again\nPress (C or c) to change directory\nPress (Any Key) To Exit\nchoice:").lower()
        match user:
            case "r":
                process.convertPictures()
            case "c":
                process.getUserDir()
                process.convertPictures()
            case _:
                break
                

