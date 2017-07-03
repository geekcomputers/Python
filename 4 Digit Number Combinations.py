#ALL the combinations of 4 digit combo
def FourDigitCombinations():
    numbers=[]
    for code in range(10000):
        if code<=9:
            code=str(code)
            numbers.append(code.zfill(4))
        elif code>=10 and code<=99:
            code=str(code)
            numbers.append(code.zfill(4))
        elif code>=100 and code<=999:
            code=str(code)
            numbers.append(code.zfill(4))
        else:
            numbers.append(str(code))
            
    for i in numbers:
        print i,   
       
    pass
                

