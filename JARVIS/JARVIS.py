#########

__author__ = "Mohammed Shokr <mohammedshokr2014@gmail.com>"
__version__ = "v 0.1"

"""
JARVIS:
- Control windows programs with your voice
"""

# import modules

import datetime
import subprocess
import webbrowser
import speech_recognition as sr
import pyttsx3
import pyjokes
from PIL import ImageGrab


class VoiceEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty("voices")
        self.engine.setProperty("voice", voices[0].id)
        self.engine.setProperty("rate", 150)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()


class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def listen(self):
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.pause_threshold = 1
            audio = self.recognizer.listen(source)

        try:
            print("Recognizing...")
            query = self.recognizer.recognize_google(audio, language="en-in")
            print(f"User said: {query}")
            return query.lower()

        except Exception:
            print("Could not understand.")
            return ""


class SystemTasks:
    def open_notepad(self):
        subprocess.call(["Notepad.exe"])

    def open_calculator(self):
        subprocess.call(["calc.exe"])

    def open_cmd(self):
        subprocess.call(["cmd.exe"])

    def open_paint(self):
        subprocess.call(["mspaint.exe"])

    def open_youtube(self):
        webbrowser.open("https://youtube.com")

    def open_google(self):
        webbrowser.open("https://google.com")

    def open_github(self):
        webbrowser.open("https://github.com")

    def take_screenshot(self):
        image = ImageGrab.grab()
        filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        image.save(filename)
        print("Screenshot saved")


class Jarvis:
    def __init__(self):
        self.voice = VoiceEngine()
        self.listener = SpeechRecognizer()
        self.tasks = SystemTasks()
        self.running = True

    def wish_me(self):
        hour = datetime.datetime.now().hour

        if hour < 12:
            self.voice.speak("Good Morning")

        elif hour < 18:
            self.voice.speak("Good Afternoon")

        else:
            self.voice.speak("Good Evening")

        self.voice.speak("I am Jarvis. How can I help you?")

    def execute_command(self, command):

        if "open notepad" in command:
            self.tasks.open_notepad()

        elif "open calculator" in command:
            self.tasks.open_calculator()

        elif "open cmd" in command:
            self.tasks.open_cmd()

        elif "open paint" in command:
            self.tasks.open_paint()

        elif "open youtube" in command:
            self.tasks.open_youtube()

        elif "open google" in command:
            self.tasks.open_google()

        elif "open github" in command:
            self.tasks.open_github()

        elif "screenshot" in command:
            self.tasks.take_screenshot()
            self.voice.speak("Screenshot taken")

        elif "joke" in command:
            joke = pyjokes.get_joke()
            print(joke)
            self.voice.speak(joke)

        elif "time" in command:
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            print(current_time)
            self.voice.speak(current_time)

        elif "exit" in command:
            self.voice.speak("Goodbye")
            self.running = False

        else:
            self.voice.speak("Command not found")

    def run(self):
        self.wish_me()

        while self.running:
            command = self.listener.listen()

            if command:
                self.execute_command(command)


if __name__ == "__main__":
    jarvis = Jarvis()
    jarvis.run()
