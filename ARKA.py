def sumOfSeries(n):
    x = n * (n + 1) // 2    #// and return x * x: Keeps the calculation in exact integers with zero precision loss
    return (x * x)


# Driver Function
n = int(input("Enter a number: "))
print(sumOfSeries(n))


#SOURCE
'''
THIS CODE IS TO FIND SUM OF CUBES OF FIRST n NATURAL NUMBERS. Formula:

∑ k³ = [ n(n + 1) / 2 ]²   (for k = 1 to n)
ex:
1³ + 2³ + ... + n³ = (n(n + 1) / 2)²

'''
