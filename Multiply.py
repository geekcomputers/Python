def product(a, b):
    # for negative vals, return the negative result of its positive eval
    if b < 0:
        return -product(a, -b)
    
    if a < b:
        return product(b, a)
    elif b != 0:
        return a + product(a, b - 1)
    else:
        return 0


a = int(input("Enter first number: "))
b = int(input("Enter second number: "))
print("Product is:", product(a, b))
