# Same as above but more pythonic
def oneLineCombinations():
    """ print out all 4-digit numbers """
    numbers = [str(i).zfill(4) for i in range(10000)]
    print(numbers)
