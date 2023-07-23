from gtts import gTTS
from playsound import playsound
import win32com
from win32com import client
import os

def tts():
    audio = 'speech.mp3'
    language = 'en'
    sentence = input("Enter the text to be spoken :- ")
    
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    sp = speaker.Speak(sentence)
    

