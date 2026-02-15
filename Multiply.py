def product(a, b):
    # Handle negative values
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
