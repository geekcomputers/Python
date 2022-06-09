from gtts import gTTS
import os

# Specify the language in which you want your audio
language = "en"

# Change the file name with the name of your txt file below:
myfile = 'text.txt'
with open(myfile, 'r') as f:
    text_from_file = f.read()
    f.close()
    
# Create an instance of gTTS class
myobj = gTTS(text=text_from_file, lang = language, slow=False)

# Method to create your audio file in mp3 format
myobj.save("hello_world.mp3")
print("Audio Saved")

# This will play your audio file
os.system("mpg321 hello_world.mp3")
