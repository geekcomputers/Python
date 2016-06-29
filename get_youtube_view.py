import time
import webbrowser

#how much views you want

totalBreaks = 30
countBreaks = 0

print("Enjoy your Time\n" +time.ctime())
while(countBreaks < totalBreaks):
    time.sleep(5) 
    webbrowser.open("https://www.youtube.com/watch?v=o6A7nf3IeeA")
    countBreaks += 1
