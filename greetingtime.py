import time
a=time.strftime('%H:%M:%S')

if a>('00:00:00') and a<('11:59:59'):
    print("good morning")
elif a>('12:00:00') and a<('17:59:59'):
    print("good afternoon")
elif a>('18:00:00') and a<('19:30:59'):
    print("good evening")
elif a>('19:31:00') and a<('24:59:59'):
    print("good night")
else:
    print("sorry")