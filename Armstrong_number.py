#checking for armstrong number
a=input('Enter a number')
n=int(a)
S=0
while n>0:
    d=n%10
    S=S+d*d*d
    n=n/10
if int(a)==S:
    print("Armstrong Number")
else:
    print("Not an Armstrong Number")
