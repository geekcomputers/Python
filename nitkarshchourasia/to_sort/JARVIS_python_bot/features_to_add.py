# imports modules
import sys
import time
from getpass import getuser

# user puts in their name
name = getuser()
name_check = input("Is your name " + name + "? → ")
if name_check.lower().startswith("y"):
    print("Okay.")
    time.sleep(1)

if name_check.lower().startswith("n"):
    name = input("Then what is it? → ")

# Can add this feature to the Jarvis.
