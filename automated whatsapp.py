import pywhatkit as kit
import time
import pyautogui
import keyboard
kit.sendwhatmsg("+91XXXXXXXXXX","hello this is an automated msg",hh,mm,ss)
pyautogui.press("enter")
time.sleep(2)
keyboard.press_and_release("enter")
print("sent")

