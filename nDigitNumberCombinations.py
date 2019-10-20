# ALL the combinations of n digit combo
def nDigitCombinations(n):
    try:
        npow = 10 ** n
        numbers = []
        for code in range(npow):
            code = str(code).zfill(n)
            numbers.append(code)
    except:
        # handle all other exceptions
        pass
    return (numbers)

# An alternate solution:
# from itertools import product
# from string import digits
# list("".join(x) for x in product(digits, repeat=n))
