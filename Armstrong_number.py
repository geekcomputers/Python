# checking for armstrong number
a = input("Enter a number")
n = int(a)
S = 0
while n > 0:
    d = n % 10
    S = S + (d**3)
    # There was an error here, it is not n = n / 10, it is n = n // 10
    n = n // 10
if int(a) == S:
    print("Armstrong Number")
else:
    print("Not an Armstrong Number")
