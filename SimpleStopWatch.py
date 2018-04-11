#Author: OMKAR PATHAK
#This script helps to build a simple stopwatch application using Python's time module.

import time

while True:
    try:
        input("Press ENTER to begin, Press Ctrl + C to stop") # For ENTER. Use raw_input() if you are running python 2.x instead of input()
        starttime = time.time()
        print('Started')
    except KeyboardInterrupt:
        print('Stopped')
        endtime = time.time()
        print('Total Time:', round(endtime - starttime, 2),'secs')
        break
