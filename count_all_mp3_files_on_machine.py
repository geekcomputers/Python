__author__ = 'tusharsappal'


import os

## This script fetches the count of the total number of mp3 [Matter of fact you can change the type of the files to be fetched by changing the endswith type]files in the listed directory

def count_all_mp3_files_on_machine():
    count = 0
    for (dirname,dirs,files) in os.walk('Replace the directory path where to serach like C:/ or / on UNIX like machines'):
        for filename in files:
            if filename.endswith(".mp3"):
                count =count+1
                thefile = os.path.join(dirname,filename)
                print "The Name is ",thefile, "and the size is ", os.path.getsize(thefile)





    print "The total number of mp3 files on the system are ",count


count_all_mp3_files_on_machine()