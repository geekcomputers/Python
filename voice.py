# modules for use of voice 
from gtts import gTTS
from colorama import Fore
import os

# Define the text you want to convert to speech
text = "Hello! This is a sample text to convert to speech."

# Exception Handaled.
try:
    # Create a gTTS object
    tts = gTTS(text=text, lang="en")

    # Save the audio file in mp3 format
    tts.save("output.mp3")

    # Play the audio file from system
    os.system("start output.mp3")
except Exception as e:
    print(Fore.RED, e, Fore.RESET)