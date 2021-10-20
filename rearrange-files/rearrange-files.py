# author : Avee Chakraborty
# department of software engineering, Daffodil inernational University
# Bangladesh

import os
import shutil


class RearrangeFile(object):
    def __init__(self):
        self.folder_path = os.getcwd()
        self.list_of_all_files = os.listdir(self.folder_path)

    def make_folder_and_return_name(self, foldername):
        if os.path.exists(foldername) is False:
            os.mkdir(foldername)
        else:
            foldername = foldername + str(2)
            os.mkdir(foldername)
        return foldername

    def check_folder_existance(self):
        for i in range(len(self.list_of_all_files)):
            if self.list_of_all_files[i].endswith('.pdf'):
                if os.path.exists('pdfs'):
                    shutil.move(self.folder_path + '/' + self.list_of_all_files[i], self.folder_path + '/pdfs')
                else:
                    os.mkdir('pdfs')

            elif self.list_of_all_files[i].endswith('jpg'):
                if os.path.exists('jpgs'):
                    shutil.move(self.folder_path + '/' + self.list_of_all_files[i], self.folder_path + '/jpgs')
                else:
                    os.mkdir('jpgs')


if __name__ == "__main__":
    re = RearrangeFile()
    re.check_folder_existance()
