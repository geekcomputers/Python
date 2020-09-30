import pyttsx3
import time
from selenium import webdriver
tts= pyttsx3.init()
rate = tts.getProperty('rate')
newVoiceRate = 160
tts.setProperty('rate', newVoiceRate)
def welcome():
    print('>')
    print("Welcome to Autobot created by Vijay.Use exit or quite to exit.")
    text="Welcome to Autobot created by Vijay"
    speak(text)
    time.sleep(1)
    text="Use exit or quite to exit."
    speak(text)
    print('<')

def speak(text):
    tts.say(text)
    tts.runAndWait()

welcome()
t=1
while(t==1):
    text=str(input(">>"))
    if 'hello' in text:
        text="hello my name is Autobot"
        print("hello my name is Autobot")
        speak(text)
        text="I can autologin to your social sites like facebook twitter github and instagram"
        print("I can autologin to your social sites like facebook twitter github and instagram")
        speak(text)
        continue
    if "facebook" or "fb" in text:
        print("Opening Your Facebook Account")
        text="Opening Your Facebook Account"
        speak(text)
        #your username and password here
        username="your username"
        password="yourpassword"
        #download webdriver of suitable version by link below
        #https://sites.google.com/a/chromium.org/chromedriver/downloads
        #locate your driver
        driver = webdriver.Chrome("C:\\Users\\AJAY\\Desktop\\chromedriver.exe")
        url="https://www.facebook.com"
        print("Opening facebook...")
        driver.get(url)
        driver.find_element_by_id('email').send_keys(username)
        print("Entering Your Username...")
        time.sleep(1)
        driver.find_element_by_id('pass').send_keys(password)
        print("Entering Your password...")
        driver.find_element_by_name('login').click()
        time.sleep(4)
        print("Login Successful")
        text="Login Successful Enjoy your day sir"
        speak(text)
        continue
    else:
        print("input valid statement")
        continue
