import time
import webbrowser

#how much views you want
views = 30

print("Enjoy your Time\n" +time.ctime())
for count in range(views):
    time.sleep(5) 
    webbrowser.open("https://www.youtube.com/watch?v=o6A7nf3IeeA")
