# Python code to print the following alphabet pattern
#A 
#B B 
#C C C 
#D D D D 
#E E E E E   
def alphabetpattern(n):
    num = 65
    for i in range(0, n):
        for j in range(0, i+1):
            ch = chr(num)
            print(ch, end=" ")
        num = num + 1
        print("\r")

a = 5
alphabetpattern(a)
