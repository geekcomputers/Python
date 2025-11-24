from gtts import gTTS
from colorama import Fore
import os

# Define the text you want to convert to speech
text = "Hello! This is a sample text to convert to speech."

try:
    # Create a gTTS object
    tts = gTTS(text=text, lang="en")

    # Save the audio file
    tts.save("output.mp3")

    # Play the audio file
    os.system("start output.mp3")
except Exception as e:
    print(Fore.RED, e, Fore.RESET)