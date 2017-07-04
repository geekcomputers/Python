# ALL the combinations of 4 digit combo
def FourDigitCombinations():
    for code in range(10000):
        print(str(code).zfill(4))
