"""
Shaurya Pratap Singh 
@shaurya-blip

Shows loading message while doing something.
"""

import itertools
import threading
import time
import sys

# The task is not done right now
done = False


def animate(message="loading", endmessage="Done!"):
    for c in itertools.cycle(["|", "/", "-", "\\"]):
        if done:
            break
        sys.stdout.write(f"\r {message}" + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write(f"\r {endmessage} ")


t = threading.Thread(
    target=lambda: animate(message="installing..", endmessage="Installation is done!!!")
)
t.start()

# Code which you are running

"""
program.install()
"""

time.sleep(10)

# Then mark done as true and thus it will end the loading screen.
done = True
