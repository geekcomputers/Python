import time

import speech_recognition as sr
import os
import playsound
import shutil

shutil.rmtree("spoken")
os.mkdir("spoken")

speeches = []


def callback(recognizer, audio):
    with open("spoken/" + str(len(speeches)) + ".wav", "wb") as file:
        file.write(audio.get_wav_data())

    playsound.playsound("spoken/" + str(len(speeches)) + ".wav")
    speeches.append(1)
    print("____")


r = sr.Recognizer()
m = sr.Microphone()
with m as source:
    r.adjust_for_ambient_noise(source)

stop_listening = r.listen_in_background(m, callback)
print("say:")
while True:
    time.sleep(0.1)
