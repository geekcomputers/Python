# ALL the combinations of 4 digit combo
def FourDigitCombinations():
    numbers = []
    for code in range(10000):
        numbers.append(str(code).zfill(4))

    for i in numbers:
        print(i)
