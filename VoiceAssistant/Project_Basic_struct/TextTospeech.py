import win32com


def tts():
    sentence = input("Enter the text to be spoken :- ")

    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    speaker.Speak(sentence)
