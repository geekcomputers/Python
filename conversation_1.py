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

# Python lists their name
userList = name

# Python & user dialoge
print("Hello", name + ", my name is Python.")
time.sleep(0.8)
print("The first letter of your name is", userList[0] + ".")
time.sleep(0.8)
print("Nice to meet you. :)")
time.sleep(0.8)
response = input("Would you say it's nice to meet me? → ")

# other dialoge
if response.lower().startswith("y"):
    print("Nice :)")
    sys.exit()

elif response.lower().startswith("n"):
    response2 = input("Is it because I am a robot? → ")

else:
    print("You may have made an input error. Please restart and try again.")
    sys.exit()
if response2.lower().startswith("y"):
    print("Aw :(")

elif response2.lower().startswith("n"):
    response3 = input("Then why? → ")
    time.sleep(1)
    print("Oh.")

else:
    print("You may have made an input error. Please restart and try again.")
    sys.exit()
