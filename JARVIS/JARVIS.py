__author__ = 'Mohammed Shokr <mohammedshokr2014@gmail.com>'
__version__ = 'v 0.1'

"""
JARVIS:
- Control windows programs with your voice
"""

# import modules
from datetime import datetime  # datetime module supplies classes for manipulating dates and times
import subprocess  # subprocess module allows you to spawn new processes

import speech_recognition as sr  # speech_recognition Library for performing speech recognition with support for Google Speech Recognition, etc..

# pip install pyttsx3                   # need to run only once to install the library

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

# recognize speech using Google Speech Recognition
Query = r.recognize_google(audio, language = 'en-IN', show_all = True )
print(Query)


# Run Application with Voice Command Function
class Jarvis:
    def __init__(self, Q):
        self.query = Q

    def sub_call(self, exe_file):
        """
        This method can directly use call method of subprocess module and according to the
        argument(exe_file) passed it returns the output.
        
        exe_file:- must pass the exe file name as str object type.
        
        """
        return subprocess.call([exe_file])

    def get_dict(self):
        '''
        This method returns the dictionary of important task that can be performed by the
        JARVIS module.
        
        Later on this can also be used by the user itself to add or update their preferred apps.
        '''
        _dict = dict(
            time=datetime.now(),
            notepad='Notepad.exe',
            calculator='calc.exe',
            stickynot='StickyNot.exe',
            shell='powershell.exe',
            paint='mspaint.exe',
            cmd='cmd.exe',
            browser='C:\Program Files\Internet Explorer\iexplore.exe',
        )
        return _dict

    @property
    def get_app(self):
        task_dict = self.get_dict()
        task = task_dict.get(self.query, None)
        if task is None:
            engine.say("Sorry Try Again")
            engine.runAndWait()
        else:
            if 'exe' in str(task):
                return self.sub_call(task)
            print(task)
            return


# Call get_app(Query) Func.
Jarvis(Query).get_app
