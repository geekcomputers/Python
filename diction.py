from difflib import get_close_matches
import pyttsx3
import json
import speech_recognition as sr

data = json.load(open("data.json"))
engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[0].id)


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        r.energy_threshold = 494
        r.adjust_for_ambient_noise(source, duration=1.5)
        audio = r.listen(source)

    try:
        print("Recognizing..")
        query = r.recognize_google(audio, language="en-in")
        print(f"User said: {query}\n")

    except Exception as e:
        # print(e)

        print("Say that again please...")
        return "None"
    return query


def translate(word):
    word = word.lower()
    if word in data:
        speak("Here is what I found in dictionary..")
        d = data[word]
        d = "".join(str(e) for e in d)
        speak(d)
    elif len(get_close_matches(word, data.keys())) > 0:
        x = get_close_matches(word, data.keys())[0]
        speak("Did you mean " + x + " instead,  respond with Yes or No.")
        ans = takeCommand().lower()
        if "yes" in ans:
            speak("ok " + "It means.." + data[x])
        elif "no" in ans:
            speak("Word doesn't exist. Please make sure you spelled it correctly.")
        else:
            speak("We didn't understand your entry.")

    else:
        speak("Word doesn't exist. Please double check it.")


if __name__ == "__main__":
    translate()
