import time
import datetime
t1=datetime.datetime.now()

print(f"Programme started at {t1}.Created by George Rahul 28/10/2020.")
z = int(input("Enter the number till which you want to find the pallindrome: "))

print("The pallindromes if any are:")
for i in range(0, z + 1):
    x = i
    y = x
    rev = 0
    while x != 0:
        d = x % 10  # takes the reminder
        rev = rev * 10 + d
        x = x // 10  # takes the quotient

    if y == rev:
        print(rev)

t2=datetime.datetime.now()

tf=t2-t1
print("\a")
print(f"It took {tf} this much time as it ended on {t2}")
time.sleep(10)