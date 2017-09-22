# ALL the combinations of 4 digit combo
def FourDigitCombinations():
    numbers=[]
    for code in range(10000):
        code=str(code).zfill(4)
        print code,
        numbers.append(code)

# Same as above but more pythonic
def oneLineCombinations():
    numbers = list(map(lambda x: str(x).zfill(4), [i for i in range(1000)]))
    print(numbers)
