__author__  = 'Mohammed Shokr <mohammedshokr2014@gmail.com>'
__version__ = 'v 0.1'

"""
JARVIS:
- Control windows programs with your voice
"""

# import modules
from datetime import datetime          # datetime module supplies classes for manipulating dates and times
import subprocess                      # subprocess module allows you to spawn new processes

import speech_recognition as sr        # speech_recognition Library for performing speech recognition with support for Google Speech Recognition, etc..



# obtain audio from the microphone
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)

# recognize speech using Google Speech Recognition
Query = r.recognize_google(audio)
print(Query)


# Run Application with Voice Command Function
def get_app(Q):
    cmd = {
        'browser': 'C:\Program Files\Internet Explorer\iexplore.exe',
        'calculator': 'calc.exe',
        'cmd': 'cmd.exe',
        'notepad': 'Notepad.exe',
        'paint': 'mspaint.exe',
        'shell': 'pwoershell.exe',
        'stikynot': 'StikyNot.exe',        
    }.get(Q)
    if cmd:
        subprocess.call([cmd])
    else:
        print(datetime.now() if Q == "time" else "Sorry!  Try again.")


# Call get_app(Query) Func.
get_app(Query)
