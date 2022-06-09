## Text To Audio

### We are using gtts module for conversion

Requirements: pip install gtts

#### Flow

gtts(Google Text To Speech)

1. Initialise the variable for your text ("mytext" line 6 in main.py)
2. Initialise the variable for language ("language" line 9 in main.py)
3. Create an instance of gTTS class ("myobj" line 12 in main.py)
4. Call the method save() and pass the filename that you want as a parameter (line 15 in main.py)
5. Play the audio file (line 19 in main.py)

#### To transcribe the text file into audio:
1. Initialise the variable for language ("language" line 5 in text-file-to-audio.py
2. Update the file name into your file name ('text.txt' line 8 in text-file-to-audio.py) 
3. Read the contents of the file and initilize the contents into a new variable ('text_from_file = f.read()' line 10 in text-file-to-audio.py )
4. Create an instance of gTTS class ("myobj" line 14 in text-file-to-audio.py)
5. Call the method save() and pass the filename that you want as a parameter (line 17 in text-file-to-audio.py)
6. Play the audio file (line 21 in text-file-to-audio.py)

#### NOTE
If you make any changes main.py, please mention it in README.md (this file). A better documentation makes the process of development faster.

---
Author - Saumitra Jagdale
 




