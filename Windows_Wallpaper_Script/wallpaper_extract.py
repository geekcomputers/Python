import os
import shutil
import time

from PIL import Image


class Wallpaper:
    # Set Environment Variables
    username = os.environ['USERNAME']
#An Amazing Code You Will Love To Have 
    # All file urls
    file_urls = {
        "wall_src": "C:\\Users\\" + username
                    + "\\AppData\\Local\\Packages\\Microsoft.Windows.ContentDeliveryManager_cw5n1h2txyewy\\"
                    + "LocalState\\Assets\\",
        "wall_dst": os.path.dirname(os.path.abspath(__file__)) + "\\Wallpapers\\",
        "wall_mobile": os.path.dirname(os.path.abspath(__file__)) + "\\Wallpapers\\mobile\\",
        "wall_desktop": os.path.dirname(os.path.abspath(__file__)) + "\\Wallpapers\\desktop\\"
    }
    msg = '''
                DDDDD      OOOOO    NN      N  EEEEEEE
                D    D    O     O   N N     N  E
                D     D   O     O   N  N    N  E
                D     D   O     O   N   N   N  EEEE
                D     D   O     O   N    N  N  E
                D    D    O     O   N     N N  E
                DDDDD      OOOOO    N      NN  EEEEEEE
            '''

    # A method to showcase time effect
    @staticmethod
    def time_gap(string):
        print(string, end='')
        time.sleep(1)
        print(".", end='')
        time.sleep(1)
        print(".")

    # A method to import the wallpapers from src folder(dir_src)
    @staticmethod
    def copy_wallpapers():
        w = Wallpaper
        w.time_gap("Copying Wallpapers")
        # Copy All Wallpapers From Src Folder To Dest Folder
        for filename in os.listdir(w.file_urls["wall_src"]):
            shutil.copy(w.file_urls["wall_src"] + filename, w.file_urls["wall_dst"])

    # A method to Change all the Extensions
    @staticmethod
    def change_ext():
        w = Wallpaper
        w.time_gap("Changing Extensions")
        # Look into all the files in the executing folder and change extension
        for filename in os.listdir(w.file_urls["wall_dst"]):
            base_file, ext = os.path.splitext(filename)
            if ext == "":
                if not os.path.isdir(w.file_urls["wall_dst"] + filename):
                    os.rename(w.file_urls["wall_dst"] + filename,
                              w.file_urls["wall_dst"] + filename + ".jpg")

    # Remove all files Not having Wallpaper Resolution
    @staticmethod
    def extract_wall():
        w = Wallpaper
        w.time_gap("Extracting Wallpapers")
        for filename in os.listdir(w.file_urls["wall_dst"]):
            base_file, ext = os.path.splitext(filename)
            if ext == ".jpg":
                try:
                    im = Image.open(w.file_urls["wall_dst"] + filename)
                except IOError:
                    print("This isn't a picture.", filename)
                if list(im.size)[0] != 1920 and list(im.size)[0] != 1080:
                    im.close()
                    os.remove(w.file_urls["wall_dst"] + filename)
                else:
                    im.close()

    # Arrange the wallpapers into the corresponding folders
    @staticmethod
    def arr_desk_wallpapers():
        w = Wallpaper
        w.time_gap("Arranging Desktop wallpapers")
        for filename in os.listdir(w.file_urls["wall_dst"]):
            base_file, ext = os.path.splitext(filename)
            if ext == ".jpg":
                try:
                    im = Image.open(w.file_urls["wall_dst"] + filename)

                    if list(im.size)[0] == 1920:
                        im.close()
                        os.rename(w.file_urls["wall_dst"] + filename,
                                  w.file_urls["wall_desktop"] + filename)
                    elif list(im.size)[0] == 1080:
                        im.close()
                        os.rename(w.file_urls["wall_dst"] + filename,
                                  w.file_urls["wall_mobile"] + filename)
                    else:
                        im.close()
                except FileExistsError:
                    print("File Already Exists!")
                    os.remove(w.file_urls["wall_dst"] + filename)

    @staticmethod
    def exec_all():
        w = Wallpaper
        w.copy_wallpapers()
        w.change_ext()
        w.extract_wall()
        w.arr_desk_wallpapers()
        print(w.msg)
        time.sleep(2)


wall = Wallpaper()
wall.exec_all()
