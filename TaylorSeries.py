#can find value of sin(x) where x is in radians upto n terms
import math

def taylorSeries(x,n):
    result=0
    for i in range(0,n):
        oddNo=2*i+1
        factorial=oddNo
        for j in range(1,oddNo):
            factorial*=j
        sign=(-1)**i
        result+=(x**oddNo)/(factorial)*sign
    return result

print(taylorSeries(math.radians(30),1))
