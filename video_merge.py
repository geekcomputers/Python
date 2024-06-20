
#Author: RIZWAN AHMAD  
#Application merge all videos in folder in single video in linux with mkvmerge
#Install with sudo apt install mkvmerge



#! /usr/bin/python3

import os
from subprocess import PIPE, Popen
#function for returning terminal command cret=command return
def cret(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        shell=True
    )
    return process.communicate()[0]

a=cret('grep | ls *.MP4 *.mp4 *mkv')
a=str(a.decode())
b=list(map(str,a.split('\n')))
print("Total video found: ",len(b))

exec="mkvmerge -o single_video.mkv " 

for x in b:
	exec+=x+" \+ "
	print(x)


out=cret(exec)

print(out)
print("Successfully executed: ")

