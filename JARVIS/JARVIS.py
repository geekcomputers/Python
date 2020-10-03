#########

__author__ = 'Mohammed Shokr <mohammedshokr2014@gmail.com>'
__version__ = 'v 0.1'

"""
JARVIS:
- Control windows programs with your voice
"""

# import modules
from datetime import datetime  # datetime module supplies classes for manipulating dates and times
import subprocess  # subprocess module allows you to spawn new processes
master
import pyjokes

=======
from playsound import *  #for sound output
master
import speech_recognition as sr  # speech_recognition Library for performing speech recognition with support for Google Speech Recognition, etc..

# pip install pyttsx3                  
# need to run only once to install the library

# importing the pyttsx3 library
import pyttsx3

# initialisation
engine = pyttsx3.init()

# obtain audio from the microphone
r = sr.Recognizer()
with sr.Microphone() as source:
    engine.say("Say something")
    engine.runAndWait()
    audio = r.listen(source)
    
    
#for audio output instead of print
def voice(p):
    myobj=gTTS(text=p,lang='en',slow=False)
    myobj.save('try.mp3')
    playsound('try.mp3')
    
# recognize speech using Google Speech Recognition
Query = r.recognize_google(audio, language = 'en-IN', show_all = True )
print(Query)


# Run Application with Voice Command Function
def get_app(Q):
    master
    if Q == "time":
        print(datetime.now())
        x=datetime.now()
        voice(x)
    elif Q == "notepad":
        subprocess.call(['Notepad.exe'])
    elif Q == "calculator":
        subprocess.call(['calc.exe'])
    elif Q == "stikynot":
        subprocess.call(['StikyNot.exe'])
    elif Q == "shell":
        subprocess.call(['powershell.exe'])
    elif Q == "paint":
        subprocess.call(['mspaint.exe'])
    elif Q == "cmd":
        subprocess.call(['cmd.exe'])
    elif Q == "browser":
        subprocess.call(['C:\Program Files\Internet Explorer\iexplore.exe'])
  master
    elif Q=="Take screenshot"
        snapshot=ImageGrab.grab()
            drive_letter = "C:\\"
            folder_name = r'downloaded-files'
            folder_time = datetime.datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
            extention = '.jpg'
            folder_to_save_files = drive_letter + folder_name + folder_time + extention
            snapshot.save(folder_to_save_files)
     
    elif Q=="Jokes":
        print(pyjokes.get_joke())

    else:
        engine.say("Sorry Try Again")
        engine.runAndWait()
   
=======
=======

    apps = {
    "time": datetime.now(),
    "notepad": "Notepad.exe",
    "calculator": "calc.exe",
    "stikynot": "StikyNot.exe",
    "shell": "powershell.exe",
    "paint": "mspaint.exe",
    "cmd": "cmd.exe",
    "browser": "C:\Program Files\Internet Explorer\iexplore.exe"
    }

    for app in apps:
        if app == Q.lower():
            subprocess.call([apps[app]])
            break
    master
    else:
        engine.say("Sorry Try Again")
        engine.runAndWait()
master
    return
# Call get_app(Query) Func.
get_app(Query)
