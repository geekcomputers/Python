#Playing your favorite youtube videos using python pywhatkit
import pywhatkit
url=input("Enter the title of the youtube video")
print("playing your requested video:")
pywhatkit.playonyt(url)#give the name of the video in this section
#Be careful when naming the video as it always plays
#the first video available on the search
print("Bye! enjoy")
