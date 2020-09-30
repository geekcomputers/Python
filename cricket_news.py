from bs4 import BeautifulSoup
import requests
import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


url = 'https://www.cricbuzz.com/cricket-news/latest-news'

ans = requests.get(url)

soup = BeautifulSoup(ans.content, 'html.parser')

anchors = soup.find_all('a', class_='cb-nws-hdln-ancr text-hvr-underline')
i = 1
speak('Welcome to sports news headlines!')
for anchor in anchors:
    speak(anchor.get_text())
    i+=1
    if i==11:
        break; 
    speak('Moving on next sports headline..')
speak('These all are major headlines, have a nice day SIR')
