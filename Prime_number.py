import math
while True:
    try:
        num = int(input("Enter a Number: "))
        break
    except ValueError:
        print("Invalid Input")

if num > 1:
    for i in range(2,int(math.sqrt(num))):  #Smallest Prime Factor of a Composite Number is less than or equal to Square Root of N
        if (num % i) == 0:
            print(num,"is NOT a Prime Number. It's indeed a COMPOSITE NUMBER")
            break
    else:
        print(num,"is a PRIME NUMBER ")
     
else:
    print(num,"is NOT a Prime Number")
