import time
import webbrowser

#how much views you want
#This only works when video has less than 300 views, it won't work when there are more than 300 views...
#due to youtube's policy.
print("Enjoy your Time\n" + time.ctime())
for count in range(30):
    time.sleep(5)
    webbrowser.open("https://www.youtube.com/watch?v=o6A7nf3IeeA")
