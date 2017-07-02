#ALL the combinations of 4 digit combo
def FourDigitCombinations():
    numbers=[]
    for code in range(10000):
        if code<=9:
            numbers.append(int("000"+str(code)))
        elif code>=10 and code<=99:
            numbers.append(int("00"+str(code)))
        elif code>=100 and code<=999:
            numbers.append(int("0"+str(code)))
        else:
            numbers.append(int(code))
            
    for i in numbers:
        print str(i),   
        
    pass
                
