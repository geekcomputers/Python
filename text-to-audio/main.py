from gtts import gTTS
import os

# Enter the text in string format which you want to convert to audio
mytext = "Hello World!, this audio is created using GTTS module."

# Specify the language in which you want your audio
language = "en"


# To convert the txt file into audio, uncomment the below lines.
# Change the file name with the name of your txt file below:
#myfile = 'name_of_your_txt_file.txt'
#with open(myfile, 'r') as f:
#    text_from_file = f.read()
#    f.close()
#myobj = gTTS(text=text_from_file, lang = language, slow=False)

# Create an instance of gTTS class
myobj = gTTS(text=mytext, lang=language, slow=False)

# Method to create your audio file in mp3 format
myobj.save("hello_world.mp3")
print("Audio Saved")

# This will play your audio file
os.system("mpg321 hello_world.mp3")

