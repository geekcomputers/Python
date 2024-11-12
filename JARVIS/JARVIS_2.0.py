import datetime
import subprocess
import os
import pyjokes
import requests
import json
import webbrowser
import smtplib
import openai
import base64
from gtts import gTTS
from PIL import ImageGrab
from pynput import keyboard
from playsound import playsound
import pyttsx3
import speech_recognition as sr

# Initialize TTS engine globally to avoid reinitializing multiple times
engine = pyttsx3.init()
engine.setProperty("voice", engine.getProperty("voices")[0].id)
engine.setProperty("rate", 150)

# Set the GPT-3 API key
api_key = base64.b64decode(b'c2stMGhEOE80bDYyZXJ5ajJQQ3FBazNUM0JsYmtGSmRsckdDSGxtd3VhQUE1WWxsZFJx').decode("utf-8")
openai.api_key = api_key

# Function to convert text to speech
def speak(audio):
    engine.say(audio)
    engine.runAndWait()

# Function to get top news headlines
def speak_news():
    try:
        url = "http://newsapi.org/v2/top-headlines?sources=the-times-of-india&apiKey=yourapikey"
        news = requests.get(url).text
        news_dict = json.loads(news)
        articles = news_dict.get("articles", [])
        speak("Source: The Times Of India")
        speak("Today's Headlines are..")
        for index, article in enumerate(articles):
            speak(article.get("title", "No title available"))
            if index == len(articles) - 1:
                break
            speak("Moving on to the next news headline..")
        speak("These were the top headlines. Have a nice day!")
    except Exception as e:
        speak("Sorry, I couldn't fetch the news.")

# Function to send an email
def send_email(to, content):
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login("youremail@gmail.com", "your-password-here")
        server.sendmail("youremail@gmail.com", to, content)
        server.close()
        speak("Email has been sent successfully!")
    except Exception as e:
        speak("Sorry, I couldn't send the email.")

# Function to interact with GPT-3
def ask_gpt3(question):
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"Answer the following question: {question}\n",
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7
        )
        answer = response.choices[0].text.strip()
        return answer
    except Exception as e:
        return "Sorry, I couldn't process your request."

# Function to wish the user based on the time of day
def wish_me():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Morning!")
    elif hour >= 12 and hour < 18:
        speak("Good Afternoon!")
    else:
        speak("Good Evening!")
    speak("I am Jarvis! How can I help you?")

# Function to capture user commands through voice
def take_command():
    wish_me()
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.pause_threshold = 1
        audio = recognizer.listen(source)
    try:
        print("Recognizing...")
        query = recognizer.recognize_google(audio, language="en-in")
        print(f"User said: {query}\n")
    except Exception as e:
        print("Say that again, please...")
        return "None"
    return query.lower()

# Function to handle application opening and tasks based on voice commands
def get_app(query):
    tasks = {
        "time": lambda: speak(datetime.datetime.now().strftime("%H:%M:%S")),
        "news": speak_news,
        "open notepad": lambda: subprocess.call(["Notepad.exe"]),
        "open calculator": lambda: subprocess.call(["calc.exe"]),
        "open stikynot": lambda: subprocess.call(["StikyNot.exe"]),
        "open shell": lambda: subprocess.call(["powershell.exe"]),
        "open paint": lambda: subprocess.call(["mspaint.exe"]),
        "open cmd": lambda: subprocess.call(["cmd.exe"]),
        "open discord": lambda: subprocess.call(["discord.exe"]),
        "open browser": lambda: subprocess.call(["C:\\Program Files\\Internet Explorer\\iexplore.exe"]),
        "open youtube": lambda: webbrowser.open("https://www.youtube.com/"),
        "open google": lambda: webbrowser.open("https://www.google.com/"),
        "open github": lambda: webbrowser.open("https://github.com/"),
        "email to other": send_email,
        "take screenshot": lambda: ImageGrab.grab().save(f"C:\\downloaded-files_{datetime.datetime.now().strftime('%Y-%m-%d_%I-%M-%S_%p')}.jpg"),
        "jokes": lambda: speak(pyjokes.get_joke()),
        "start recording": lambda: subprocess.call(["start", "recording"]),  # Placeholder
        "stop recording": lambda: subprocess.call(["stop", "recording"]),  # Placeholder
        "clip that": lambda: subprocess.call(["clip", "that"]),  # Placeholder
    }

    # Check if the query matches any pre-defined task
    if query in tasks:
        tasks[query]()
    elif "search for" in query:
        answer = ask_gpt3(query.replace("search for", "").strip())
        speak(answer)
    else:
        answer = ask_gpt3(query)
        speak(answer)

# Main loop to run Jarvis
def run_jarvis():
    while True:
        query = take_command()
        if query == "take a break":
            break
        get_app(query)

if __name__ == "__main__":
    run_jarvis()
