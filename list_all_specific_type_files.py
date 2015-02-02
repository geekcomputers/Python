__author__ = 'tusharsappal'
import os

## This program searches in the current working directory and lists all the python files



def print_the_list_of_python_files():
    cwd = os.getcwd()
    print "The current working directory is ", cwd
    cwd_2 = os.listdir(cwd)
    for file in cwd_2:
        if file.endswith(".py"):   ## To change the type of the file fetched change the ends with to the specific file name

            print "The pyhton file is ", file





print_the_list_of_python_files()

