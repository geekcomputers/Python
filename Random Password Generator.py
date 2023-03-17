import random

low="abcdefghijklmnopqrstuvwxyz"
upp="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
num="0123456789"
sym="!@#$%^&*"

all=low+upp+num+sym
length=8
password="".join(random.sample(all,length))
print(password)

