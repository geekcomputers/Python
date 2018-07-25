# ALL the combinations of 4 digit combo
def FourDigitCombinations():
    numbers=[]
    for code in range(10000):
        code=str(code).zfill(4)
        print(code)
        numbers.append(code)

# Same as above but more pythonic
def oneLineCombinations():    
    numbers = [str(i).zfill(4) for i in range(1000)]
    print(numbers)
