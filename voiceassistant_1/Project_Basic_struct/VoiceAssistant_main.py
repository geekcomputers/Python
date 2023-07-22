from speakListen import *
from websiteWork import *
from textRead import *
from dictator import *
from menu import *
from speechtotext import *
from TextTospeech import *


def main():
    start = 0
    end = 0
    if start == 0:
        print("\nSay \"Hello Python\" to activate the Voice Assistant!")
        start += 1
    while True:
        
        q = short_hear().lower()
        if "close" in q:
            greet("end")
            exit(0)
        if "hello python" in q:
            greet("start")
            print_menu()
            while True:
                
                query = hear().lower()
                if "close" in query:
                    greet("end")
                    end += 1
                    return 0
                elif "text to speech" in query:
                    tts()
                    time.sleep(4)
                    

                elif "search on google" in query or "search google" in query or "google" in query:
                    google_search()
                    time.sleep(10)
                    
                elif "search on wikipedia" in query or "search wikipedia" in query or "wikipedia" in query:
                    wiki_search()
                    time.sleep(10)
                    
                elif "word" in query:
                    ms_word()
                    time.sleep(5)
                    
                elif "book" in query:
                    pdf_read()
                    time.sleep(10)
                   
                elif "speech to text" in query:
                    big_text()
                    time.sleep(5)
                    
                else:
                    print("I could'nt understand what you just said!")
                    speak("I could'nt understand what you just said!")
                
                print("\nDo you want to continue? if yes then say " + Fore.YELLOW + "\"YES\"" + Fore.WHITE + " else say " + Fore.YELLOW + "\"CLOSE PYTHON\"")
                speak("Do you want to continue? if yes then say YES else say CLOSE PYTHON")
                qry = hear().lower()
                if "yes" in qry:
                    print_menu()
                elif "close" in qry:
                    greet("end")
                    return 0
                else:
                    speak("You didn't say a valid command. So I am continuing!")
                    continue

        elif "close" in q:
            return 0
        else:
            continue

main()
