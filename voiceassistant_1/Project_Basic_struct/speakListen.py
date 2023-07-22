import time
from colorama import Fore, Back, Style
import speech_recognition as sr
import os
import pyttsx3
import datetime
from rich.progress import Progress


python = pyttsx3.init("sapi5") # name of the engine is set as Python
voices = python.getProperty("voices")
#print(voices)
python.setProperty("voice", voices[1].id)
python.setProperty("rate", 140)


def speak(text):
    """[This function would speak aloud some text provided as parameter]

    Args:
        text ([str]): [It is the speech to be spoken]
    """    
    python.say(text)
    python.runAndWait()

def greet(g):
    """Uses the datetime library to generate current time and then greets accordingly.
    

    Args:
        g (str): To decide whether to say hello or good bye
    """
    if g == "start" or g == "s":
        h = datetime.datetime.now().hour
        text = ''
        if h > 12 and h < 17:
            text = "Hello ! Good Afternoon  "
        elif h < 12 and h > 0:
            text = "Hello! Good Morning  "
        elif h >= 17 :
            text = "Hello! Good Evening "
        text += " I am Python, How may i help you ?"
        speak(text)    
    
    elif g == "quit" or g == "end" or g == "over" or g == "e":
        text = 'Thank you!. Good Bye ! '
        speak(text)

def hear():
    """[It will process the speech of user using Google_Speech_Recognizer(recognize_google)]

    Returns:
        [str]: [Speech of user as a string in English(en - IN)]
    """    
    r = sr.Recognizer()
    """Reconizer is a class which has lot of functions related to Speech i/p and o/p.
    """
    r.pause_threshold = 1 # a pause of more than 1 second will stop the microphone temporarily
    r.energy_threshold = 300 # python by default sets it to 300. It is the minimum input energy to be considered. 
    r.dynamic_energy_threshold = True # pyhton now can dynamically change the threshold energy

    with sr.Microphone() as source:
        # read the audio data from the default microphone
        print(Fore.RED + "\nListening...")
        #time.sleep(0.5)

        speech = r.record(source, duration = 9)  # option 
        #speech = r.listen(source)
        # convert speech to text
        try:
            #print("Recognizing...")
            recognizing()
            speech = r.recognize_google(speech)
            print(speech + "\n")
        
        except Exception as exception:
            print(exception)
            return "None"
    return speech

def recognizing():
    """Uses the Rich library to print a simulates version of "recognizing" by printing a loading bar.
    """
    with Progress() as pr:
        rec = pr.add_task("[red]Recognizing...", total = 100)
        while not pr.finished:
            pr.update(rec, advance = 1.0)
            time.sleep(0.01)

def long_hear(duration_time = 60):
    """[It will process the speech of user using Google_Speech_Recognizer(recognize_google)]
        the difference between the hear() and long_hear() is that - the
        hear() - records users voice for 9 seconds
        long_hear() - will record user's voice for the time specified by user. By default, it records for 60 seconds.
    Returns:
        [str]: [Speech of user as a string in English(en - IN)]
    """    
    r = sr.Recognizer()
    """Reconizer is a class which has lot of functions related to Speech i/p and o/p.
    """
    r.pause_threshold = 1 # a pause of more than 1 second will stop the microphone temporarily
    r.energy_threshold = 300 # python by default sets it to 300. It is the minimum input energy to be considered. 
    r.dynamic_energy_threshold = True # pyhton now can dynamically change the threshold energy

    with sr.Microphone() as source:
        # read the audio data from the default microphone
        print(Fore.RED + "\nListening...")
        #time.sleep(0.5)

        speech = r.record(source, duration = duration_time)  # option 
        #speech = r.listen(source)
        # convert speech to text
        try:
            print(Fore.RED +"Recognizing...")
            #recognizing()
            speech = r.recognize_google(speech)
            #print(speech + "\n")
        
        except Exception as exception:
            print(exception)            
            return "None"
    return speech

def short_hear(duration_time = 5):
    """[It will process the speech of user using Google_Speech_Recognizer(recognize_google)]
        the difference between the hear() and long_hear() is that - the
        hear() - records users voice for 9 seconds
        long_hear - will record user's voice for the time specified by user. By default, it records for 60 seconds.
    Returns:
        [str]: [Speech of user as a string in English(en - IN)]
    """    
    r = sr.Recognizer()
    """Reconizer is a class which has lot of functions related to Speech i/p and o/p.
    """
    r.pause_threshold = 1 # a pause of more than 1 second will stop the microphone temporarily
    r.energy_threshold = 300 # python by default sets it to 300. It is the minimum input energy to be considered. 
    r.dynamic_energy_threshold = True # pyhton now can dynamically change the threshold energy

    with sr.Microphone() as source:
        # read the audio data from the default microphone
        print(Fore.RED + "\nListening...")
        #time.sleep(0.5)

        speech = r.record(source, duration = duration_time)  # option 
        #speech = r.listen(source)
        # convert speech to text
        try:
            print(Fore.RED +"Recognizing...")
            #recognizing()
            speech = r.recognize_google(speech)
            #print(speech + "\n")
        
        except Exception as exception:
            print(exception)            
            return "None"
    return speech

        

if __name__ == '__main__':
    # print("Enter your name")
    # name = hear()
    # speak("Hello " + name)
    # greet("s")
    # greet("e")
    pass
    #hear()
    #recognizing()
    
