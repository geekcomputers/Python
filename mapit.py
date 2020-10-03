import sys,webbrowser,pyperclip
if len(sys.argv)>1:
    address = ' '.join(sys.argv[1:])

elif len(pyperclip.paste())> 2:
    address = pyperclip.paste()
else:
    address = input("enter your palce")
webbrowser.open('https://www.google.com/maps/place/'+address)
