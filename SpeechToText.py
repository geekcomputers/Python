import pyttsx3

engine = pyttsx3.init()

voices = engine.getProperty("voices")
for voice in voices:
    print(voice.id)
    print(voice.name)

id = r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0"
engine.setProperty("voices", id)
engine.setProperty("rate", 165)
engine.say("jarivs")  # Replace string with our own text
engine.runAndWait()
