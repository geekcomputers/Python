
from gtts import gTTS 
import os 

# Enter the text in string format which you want to convert to audio
mytext = "Hello World!, this audio is created using GTTS module."

# Specify the language in which you want your audio
language = 'en'

# Create an instance of gTTS class 
myobj = gTTS(text=mytext, lang=language, slow=False) 

# Method to create your audio file in mp3 format
myobj.save("hello_world.mp3") 
print("Audio Saved")

# This will play your audio file
os.system("mpg321 welcome.mp3") 
