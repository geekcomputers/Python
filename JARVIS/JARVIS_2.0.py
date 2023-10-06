#########

__author__ = "Mohammed Shokr <mohammedshokr2014@gmail.com>"
__version__ = "v 0.1"

"""
JARVIS:
- Control windows programs with your voice
"""

# import modules
import datetime  # datetime module supplies classes for manipulating dates and times
import subprocess  # subprocess module allows you to spawn new processes

# master
import pyjokes # for generating random jokes
import requests
import json
from PIL import Image, ImageGrab
from gtts import gTTS

# for 30 seconds clip "Jarvis, clip that!" and discord ctrl+k quick-move (might not come to fruition)
from pynput import keyboard
from pynput.keyboard import Key, Listener
from pynput.mouse import Button, Controller

# =======
from playsound import *  # for sound output

# master
# auto install for pyttsx3 and speechRecognition
import os
try:
    import pyttsx3 #Check if already installed
except:# If not installed give exception
    os.system('pip install pyttsx3')#install at run time
    import pyttsx3 #import again for speak function

try :
    import speech_recognition as sr
except:
    os.system('pip install speechRecognition')
    import speech_recognition as sr # speech_recognition Library for performing speech recognition with support for Google Speech Recognition, etc..

# importing the pyttsx3 library
import webbrowser
import smtplib

# initialisation
engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[0].id)
engine.setProperty("rate", 150)
exit_jarvis = False


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def speak_news():
    url = "http://newsapi.org/v2/top-headlines?sources=the-times-of-india&apiKey=yourapikey"
    news = requests.get(url).text
    news_dict = json.loads(news)
    arts = news_dict["articles"]
    speak("Source: The Times Of India")
    speak("Todays Headlines are..")
    for index, articles in enumerate(arts):
        speak(articles["title"])
        if index == len(arts) - 1:
            break
        speak("Moving on the next news headline..")
    speak("These were the top headlines, Have a nice day Sir!!..")


def sendEmail(to, content):
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.ehlo()
    server.starttls()
    server.login("youremail@gmail.com", "yourr-password-here")
    server.sendmail("youremail@gmail.com", to, content)
    server.close()

import openai
import base64 
stab=(base64.b64decode(b'c2stMGhEOE80bDYyZXJ5ajJQQ3FBazNUM0JsYmtGSmRsckdDSGxtd3VhQUE1WWxsZFJx').decode("utf-8"))
api_key = stab
def ask_gpt3(que):
    openai.api_key = api_key

    response = openai.Completion.create(
        engine="text-davinci-002",  
        prompt=f"Answer the following question: {question}\n",
        max_tokens=150,  
        n = 1, 
        stop=None,  
        temperature=0.7  
    )

    answer = response.choices[0].text.strip()
    return answer

def wishme():
    # This function wishes user
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Morning!")
    elif hour >= 12 and hour < 18:
        speak("Good Afternoon!")
    else:
        speak("Good Evening!")
    speak("I m Jarvis  ! how can I help you sir")


# obtain audio from the microphone
def takecommand():
    # it takes user's command and returns string output
    wishme()
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        r.dynamic_energy_threshold = 500
        audio = r.listen(source)
    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language="en-in")
        print(f"User said {query}\n")
    except Exception as e:
        print("Say that again please...")
        return "None"
    return query


# for audio output instead of print
def voice(p):
    myobj = gTTS(text=p, lang="en", slow=False)
    myobj.save("try.mp3")
    playsound("try.mp3")


# recognize speech using Google Speech Recognition


def on_press(key):
    if key == keyboard.Key.esc:
        return False  # stop listener
    try:
        k = key.char  # single-char keys
    except:
        k = key.name  # other keys
    if k in ["1", "2", "left", "right"]:  # keys of interest
        # self.keys.append(k)  # store it in global-like variable
        print("Key pressed: " + k)
        return False  # stop listener; remove this if want more keys


# Run Application with Voice Command Function
# only_jarvis
def on_release(key):
    print("{0} release".format(key))
    if key == Key.esc():
        # Stop listener
        return False
    """
class Jarvis:
    def __init__(self, Q):
        self.query = Q

    def sub_call(self, exe_file):
        '''
        This method can directly use call method of subprocess module and according to the
        argument(exe_file) passed it returns the output.

        exe_file:- must pass the exe file name as str object type.

        '''
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
            browser='C:\\Program Files\\Internet Explorer\\iexplore.exe',
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


# =======
"""


def get_app(Q):
    current = Controller()
    # master
    if Q == "time":
        print(datetime.now())
        x = datetime.now()
        voice(x)
    elif Q == "news":
        speak_news()

    elif Q == "open notepad":
        subprocess.call(["Notepad.exe"])
    elif Q == "open calculator":
        subprocess.call(["calc.exe"])
    elif Q == "open stikynot":
        subprocess.call(["StikyNot.exe"])
    elif Q == "open shell":
        subprocess.call(["powershell.exe"])
    elif Q == "open paint":
        subprocess.call(["mspaint.exe"])
    elif Q == "open cmd":
        subprocess.call(["cmd.exe"])
    elif Q == "open discord":
        subprocess.call(["discord.exe"])
    elif Q == "open browser":
        subprocess.call(["C:\\Program Files\\Internet Explorer\\iexplore.exe"])
    # patch-1
    elif Q == "open youtube":
        webbrowser.open("https://www.youtube.com/")  # open youtube
    elif Q == "open google":
        webbrowser.open("https://www.google.com/")  # open google
    elif Q == "open github":
        webbrowser.open("https://github.com/")
    elif Q == "search for":
        que=Q.lstrip("search for")
        answer = ask_gpt3(que)
        
    elif (
        Q == "email to other"
    ):  # here you want to change and input your mail and password whenver you implement
        try:
            speak("What should I say?")
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("Listening...")
                r.pause_threshold = 1
                audio = r.listen(source)
            to = "abc@gmail.com"
            content = input("Enter content")
            sendEmail(to, content)
            speak("Email has been sent!")
        except Exception as e:
            print(e)
            speak("Sorry, I can't send the email.")
    # =======
    #   master
    elif Q == "Take screenshot":
        snapshot = ImageGrab.grab()
        drive_letter = "C:\\"
        folder_name = r"downloaded-files"
        folder_time = datetime.datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
        extention = ".jpg"
        folder_to_save_files = drive_letter + folder_name + folder_time + extention
        snapshot.save(folder_to_save_files)

    elif Q == "Jokes":
        speak(pyjokes.get_joke())

    elif Q == "start recording":
        current.add("Win", "Alt", "r")
        speak("Started recording. just say stop recording to stop.")

    elif Q == "stop recording":
        current.add("Win", "Alt", "r")
        speak("Stopped recording. check your game bar folder for the video")

    elif Q == "clip that":
        current.add("Win", "Alt", "g")
        speak("Clipped. check you game bar file for the video")
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    elif Q == "take a break":
        exit()
    else:
        answer = ask_gpt3(Q)

    # master

    apps = {
        "time": datetime.datetime.now(),
        "notepad": "Notepad.exe",
        "calculator": "calc.exe",
        "stikynot": "StikyNot.exe",
        "shell": "powershell.exe",
        "paint": "mspaint.exe",
        "cmd": "cmd.exe",
        "browser": "C:\\Program Files\Internet Explorer\iexplore.exe",
        "vscode": "C:\\Users\\Users\\User\\AppData\\Local\\Programs\Microsoft VS Code"
    }
    # master


# Call get_app(Query) Func.

if __name__ == "__main__":
    while not exit_jarvis:
        Query = takecommand().lower()
        get_app(Query)
    exit_jarvis = True
