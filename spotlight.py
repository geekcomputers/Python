""" Script To Copy Spotlight(Lockscreen) Images from Windows """
from __future__ import print_function

import errno
import hashlib
import os
import shutil

from PIL import Image

try:
    input = raw_input
except NameError:
    pass


def md5(fname):
    """ Function to return the MD5 Digest of a file """

    hash_md5 = hashlib.md5()
    with open(fname, "rb") as file_var:
        for chunk in iter(lambda: file_var.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def make_folder(folder_name):
    """Function to make the required folers"""
    try:
        os.makedirs(folder_name)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(folder_name):
            pass
        else:
            print("Error! Could not create a folder")
            raise


def get_spotlight_wallpapers(target_folder):
    """Fetches wallpapers from source folder inside AppData to the
    newly created folders in C:\\Users\\['user.name']\\Pictures"""
    # PATHS REQUIRED TO FETCH AND STORE WALLPAPERS
    # Creating necessary folders

    source_folder = os.environ['HOME'] + "\\AppData\\Local\\Packages\\"
    source_folder += "Microsoft.Windows.ContentDeliveryManager_cw5n1h2txyewy"
    source_folder += "\\LocalState\\Assets"
    spotlight_path_mobile = target_folder + "\\Mobile"
    spotlight_path_desktop = target_folder + "\\Desktop"
    make_folder(spotlight_path_mobile)
    make_folder(spotlight_path_desktop)

    # Fetching files from the source dir
    for filename in os.listdir(source_folder):
        filename = source_folder + "\\" + filename
        # if size of file is less than 100 KB, ignore the file
        if os.stat(filename).st_size > 100000:
            # Check resolution and classify based upon the resolution of the images

            # name the file equal to the MD5 of the file, so that no duplicate files are to be copied
            img_file = Image.open(filename)
            if img_file.size[0] >= 1080:
                if img_file.size[0] > img_file.size[1]:
                    temp_path = spotlight_path_desktop + "\\" + md5(filename)
                else:
                    temp_path = spotlight_path_mobile + "\\" + md5(filename)
                # If file doesn't exist, copy the file to the new folders
                if not os.path.exists(temp_path + ".png"):
                    shutil.copy(filename, temp_path + ".png")


if __name__ == '__main__':
    PATH = input("Enter directory path:").strip()
    get_spotlight_wallpapers(PATH)
    print("Lockscreen images have been copied to \"" + PATH + "\"")
