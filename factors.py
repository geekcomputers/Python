import math

print('The factors of the number you type when prompted will be displayed')
a = int(input('Type now // '))
b = 1
while b <= math.sqrt(a):
    if a % b == 0:
        print("A factor of the number is ", b)
        print("A factor of the number is ", int(a / b))
    b += 1
