#! /usr/bin/python3



# Author Name: RIZWAN AHMAD

# Package required: mediainfo ("sudo apt install mediainfo")

# Description: A python script to find total time for all videos in a folder

''' ("I am used for find how many time required for finsih GATE lectures in one folder so it will be for motivation that
i can finish this lectures in only 5hr now i have to sit 2-3hr in 2x speed i finished my lecture so easy")'''





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


d=list(cret('ls').decode().split('\n'))
for i in range(len(d)-1):
    print("Press ",i+1,"For ",d[i])

inp=int(input())-1
os.chdir(d[inp])

#Must install mediainfo ("sudo apt install mediainfo")

a=cret('mediainfo --Inform="Video;%Duration/String3%" *.MP4 *.mp4 *.mkv *.wmv *.avi *.3gp *.webm')
a=str(a.decode())
b=list(map(str,a.split('.')))
minute,sec=0,0
for i in range(len(b)-1):
	l=[]
	l=list(map(str,b[i].split(':')))
	minute+=int(l[1])
	sec+=int(l[2])
msec=int(sec/60)
minute+=msec
hr=minute//60
minute=minute%60
print("Total Videos :",len(b)-1,"\nTotal length :",hr,"hr",minute,"min")
a=input("Press Enter To Continue")
