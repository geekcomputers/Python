
# patch-255
decimal_accuracy = 7

def dtbconverter(num): 

    whole = [] 
    fractional = ['.']  

    decimal = round(num % 1, decimal_accuracy) 
    w_num = int(num) 

    i = 0  
    while (decimal != 1 and i < decimal_accuracy):
        decimal = decimal * 2
        fractional.append(int(decimal // 1))
        decimal = round(decimal % 1, decimal_accuracy)
        if (decimal == 0):
            break 
        i +=1

    while (w_num != 0):
        whole.append(w_num % 2)
        w_num = w_num // 2
    whole.reverse()
    
    i=0
    while(i<len(whole)):
        print(whole[i],end="")
        i+=1
    i=0
    while(i<len(fractional)):
        print(fractional[i],end="")
        i+=1
    
    
number = float(input("Enter Any base-10 Number: "))
 
dtbconverter(number)


#i think this code have not proper comment and noe this is easy to understand
'''
=======
Program: Decimal to Binary converter.

THis program accepts fractional values, the accuracy can be set below:
'''

# Function to convert decimal number 
# to binary using recursion 
def DecimalToBinary(num): 
      
    if num > 1: 
        DecimalToBinary(num // 2) 
    print(num % 2, end = '') 
  
# Driver Code 
if __name__ == '__main__': 
      
    # decimal value 
    dec_val = 24
      
    # Calling function 
    DecimalToBinary(dec_val) 
# master
