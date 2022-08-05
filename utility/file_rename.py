"""
Following script can be used to rename bulk of file at a time.
"""
import os


def rename_files(folder_path, new_name):

    for count, i in enumerate(os.listdir(folder_path)):
        print(i)
        os.rename(
            f"utility\\demo_files\\{i}", f"utility\\demo_files\\{str(count) + new_name }")


if __name__ == "__main__":

    rename_files("utility\\demo_files\\", "file.txt")
