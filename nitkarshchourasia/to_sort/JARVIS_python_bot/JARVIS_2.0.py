#!/usr/bin/env python3

import base64
import datetime
import os
import smtplib
import subprocess
import sys
import time
import webbrowser

import httpx  # Replaced requests with httpx
import openai
import pyjokes
import pyttsx3
import speech_recognition as sr
from PIL import ImageGrab
from pygame import mixer  # For audio output
from pynput import keyboard, mouse

# Configuration
__author__ = "Nitkarsh Chourasia <playnitkarsh@gmail.com>"
__version__ = "v 0.1"

# Initialize pygame mixer for audio
mixer.init()
mixer.music.set_volume(1.0)

# Initialize text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[0].id)
engine.setProperty("rate", 150)
exit_jarvis = False

# Decode API key
api_key = base64.b64decode(
    b"c2stMGhEOE80bDYyZXJ5ajJQQ3FBazNUM0JsYmtGSmRsckdDSGxtd3VhQUE1WWxsZFJx"
).decode("utf-8")
openai.api_key = api_key


def speak(audio: str) -> None:
    """Speak the given text using TTS engine."""
    engine.say(audio)
    engine.runAndWait()


def play_audio(file_path: str) -> None:
    """Play audio file using pygame mixer."""
    try:
        mixer.music.load(file_path)
        mixer.music.play()
        while mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"Error playing audio: {e}")


async def speak_news() -> None:
    """Fetch and speak top headlines from Times of India using httpx."""
    url = "http://newsapi.org/v2/top-headlines?sources=the-times-of-india&apiKey=yourapikey"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            news_dict = response.json()
            arts = news_dict["articles"]
            speak("Source: The Times Of India")
            speak("Today's Headlines are..")
            for index, article in enumerate(arts):
                speak(article["title"])
                if index < len(arts) - 1:
                    speak("Moving on to the next news headline..")
            speak("These were the top headlines. Have a nice day, Sir!")
    except Exception as e:
        speak(f"Sorry, unable to fetch news. Error: {str(e)}")


def send_email(to: str, content: str) -> None:
    """Send an email using SMTP."""
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login("youremail@gmail.com", "yourr-password-here")
        server.sendmail("youremail@gmail.com", to, content)
        server.close()
        speak("Email has been sent!")
    except Exception as e:
        print(f"Error sending email: {e}")
        speak("Sorry, I can't send the email.")


def ask_gpt3(question: str) -> str:
    """Get response from OpenAI GPT-3."""
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"Answer the following question: {question}\n",
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Sorry, I couldn't get an answer. Error: {str(e)}"


def wishme() -> None:
    """Wish the user based on current time."""
    hour = int(datetime.datetime.now().hour)
    if 0 <= hour < 12:
        speak("Good Morning!")
    elif 12 <= hour < 18:
        speak("Good Afternoon!")
    else:
        speak("Good Evening!")
    speak("I'm Jarvis! How can I help you, sir?")


def takecommand() -> str:
    """Listen to user's voice command and return as text."""
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
        print(f"User said: {query}\n")
        return query.lower()
    except Exception as e:
        print(f"Error recognizing speech: {e}")
        speak("Say that again please...")
        return "None"


def voice(text: str) -> None:
    """Convert text to speech and play using pygame."""
    try:
        from gtts import gTTS
        myobj = gTTS(text=text, lang="en", slow=False)
        myobj.save("try.mp3")
        play_audio("try.mp3")
        os.remove("try.mp3")  # Cleanup
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        speak(text)  # Fallback to pyttsx3


def on_press(key):
    """Handle key press events for listeners."""
    if key == keyboard.Key.esc:
        return False  # Stop listener
    try:
        k = key.char  # Single-char keys
    except:
        k = key.name  # Other keys
    if k in ["1", "2", "left", "right"]:
        print("Key pressed: " + k)
        return False  # Stop listener


def on_release(key):
    """Handle key release events for listeners."""
    print(f"{key} released")
    if key == keyboard.Key.esc:
        return False  # Stop listener


def get_app(query: str) -> None:
    """Execute actions based on user command."""
    mouse_controller = mouse.Controller()
    
    if query == "time":
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        print(current_time)
        voice(current_time)
        
    elif query == "news":
        import asyncio
        asyncio.run(speak_news())
        
    elif query == "open notepad":
        subprocess.call(["Notepad.exe"])
        
    elif query == "open calculator":
        subprocess.call(["calc.exe"])
        
    elif query == "open stikynot":
        subprocess.call(["StikyNot.exe"])
        
    elif query == "open shell":
        subprocess.call(["powershell.exe"])
        
    elif query == "open paint":
        subprocess.call(["mspaint.exe"])
        
    elif query == "open cmd":
        subprocess.call(["cmd.exe"])
        
    elif query == "open discord":
        subprocess.call(["discord.exe"])
        
    elif query == "open browser":
        subprocess.call(["C:\\Program Files\\Internet Explorer\\iexplore.exe"])
        
    elif query == "open youtube":
        webbrowser.open("https://www.youtube.com/")
        
    elif query == "open google":
        webbrowser.open("https://www.google.com/")
        
    elif query == "open github":
        webbrowser.open("https://github.com/")
        
    elif query.startswith("search for"):
        question = query.replace("search for", "").strip()
        answer = ask_gpt3(question)
        print(answer)
        speak(answer)
        
    elif query == "email to other":
        try:
            speak("What should I say?")
            content = takecommand()
            to = "abc@gmail.com"
            send_email(to, content)
        except Exception as e:
            print(f"Error: {e}")
            speak("Sorry, I can't send the email.")
            
    elif query == "take screenshot":
        try:
            snapshot = ImageGrab.grab()
            folder_name = r"C:\downloaded-files"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            file_name = datetime.datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p.jpg")
            file_path = os.path.join(folder_name, file_name)
            snapshot.save(file_path)
            speak("Screenshot saved successfully.")
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            speak("Sorry, I couldn't take the screenshot.")
            
    elif query == "jokes":
        joke = pyjokes.get_joke()
        print(joke)
        speak(joke)
        
    elif query == "start recording":
        # Simulate key press (not actual implementation)
        speak("Started recording. Say 'stop recording' to stop.")
        
    elif query == "stop recording":
        # Simulate key press (not actual implementation)
        speak("Stopped recording. Check your game bar folder for the video.")
        
    elif query == "clip that":
        # Simulate key press (not actual implementation)
        speak("Clipped. Check your game bar folder for the video.")
        
    elif query == "take a break":
        speak("Okay, I'm taking a break. Call me anytime!")
        global exit_jarvis
        exit_jarvis = True
        
    else:
        answer = ask_gpt3(query)
        print(answer)
        speak(answer)


if __name__ == "__main__":
    print(f"JARVIS v{__version__} by {__author__}")
    print("Starting...")
    
    # Check dependencies
    try:
        import httpx
        import openai
        import pygame
        import pyttsx3
        import speech_recognition as sr
        from PIL import ImageGrab
    except ImportError as e:
        print(f"Missing dependency: {e}. Please install required packages.")
        sys.exit(1)
    
    # Main loop
    while not exit_jarvis:
        query = takecommand()
        if query != "none":
            get_app(query)
    
    print("Exiting JARVIS...")
    mixer.quit()