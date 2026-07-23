# install and import google text-to-speech library gtts
from gtts import gTTS
import os

# provide user input text
text = input("enter the text: ")
# covert text into voice
voice = gTTS(text=text, lang="en")
# save the generated voice
voice.save("output.mp3")
# play the file in windows
os.system("start output.mp3")
