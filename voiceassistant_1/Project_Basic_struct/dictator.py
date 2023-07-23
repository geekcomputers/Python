# from speakListen import hear
# from speakListen import long_hear
from speakListen import *

from colorama import Fore, Back, Style

def big_text():
    print("By default, I will record your voice for 60 seconds.\nDo you want to change this default timing?")
    speak("By default, I will record your voice for 60 seconds.\nDo you want to change this default timing?")
    print(Fore.YELLOW + "Yes or No")
    query = hear().lower()

    duration_time = 0

    if  "yes" in query or "es" in query or "ye" in query or "s" in query:

        print("Please enter the time(in seconds) for which I shall record your speech - ", end = '')
        duration_time = int(input().strip())

        print("\n")
    else:
        duration_time = 60
    speak(f"I will record for {duration_time} seconds!")
    text = long_hear(duration_time)
    print("\n" + Fore.LIGHTCYAN_EX + text)

def colours():
    text = "Colour"
    print(Fore.BLACK + text)
    print(Fore.GREEN + text)
    print(Fore.YELLOW + text)
    print(Fore.RED + text)
    print(Fore.BLUE + text)
    print(Fore.MAGENTA + text)
    print(Fore.CYAN + text)
    print(Fore.WHITE + text)
    print(Fore.LIGHTBLACK_EX + text)
    print(Fore.LIGHTRED_EX + text)
    print(Fore.LIGHTGREEN_EX + text)
    print(Fore.LIGHTYELLOW_EX + text)
    print(Fore.LIGHTBLUE_EX + text)
    print(Fore.LIGHTMAGENTA_EX + text)
    print(Fore.LIGHTCYAN_EX + text)
    print(Fore.LIGHTWHITE_EX + text)

#big_text()
#colours()