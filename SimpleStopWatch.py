# Author: OMKAR PATHAK
# This script helps to build a simple stopwatch application using Python's time module.

import time

print('Press ENTER to begin, Press Ctrl + C to stop')
while True:
    try:
        input()  # For ENTER. Use raw_input() if you are running python 2.x instead of input()
        starttime = time.time()
        print('Started')
        while True:
            print('Time Elapsed: ', round(time.time() - starttime, 0), 'secs', end="\r")
            time.sleep(1) # 1 second delay
    except KeyboardInterrupt:
        print('Stopped')
        endtime = time.time()
        print('Total Time:', round(endtime - starttime, 2), 'secs')
        break
