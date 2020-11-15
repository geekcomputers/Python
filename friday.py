import pyttsx3
import os

var = 1

while var>0:
 pyttsx3.speak("How can I help you Sir")
 print("How can I help you Sir : ", end = '')
 x=input()
 if (("notepad" in x) or ("Notepad" in x)) and (("open" in x) or ("run" in x) or ("Open" in x) or ("Run" in x)) :
 	pyttsx3.speak("Here it is , sir")
 	os.system("notepad")
 print("anything more")
 	
