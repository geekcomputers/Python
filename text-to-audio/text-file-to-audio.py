from gtts import gTTS
import os

# Enter the name of your text file
mytextfile = "hello.txt"

# Specify the language in which you want your audio
language = "en"

# Get the contents of your file
with open(mytextfile, 'r') as f:
    mytext = f.read()
    f.close()

# Create an instance of gTTS class
myobj = gTTS(text=mytext, lang=language, slow=False)

# Method to create your audio file in mp3 format
myobj.save("hello.mp3")
print("Audio Saved")

# This will play your audio file
os.system("mpg321 hello.mp3")
