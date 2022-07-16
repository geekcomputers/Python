#Way2 1:

#isHappyNumber() will determine whether a number is happy or not    
def isHappyNumber(num):    
    rem = sum = 0;    
        
    #Calculates the sum of squares of digits    
    while(num > 0):    
        rem = num%10;    
        sum = sum + (rem*rem);    
        num = num//10;    
    return sum;    
        
num = 82;    
result = num;    
     
while(result != 1 and result != 4):    
    result = isHappyNumber(result);    
     
#Happy number always ends with 1    
if(result == 1):    
    print(str(num) + " is a happy number after apply way 1");    
#Unhappy number ends in a cycle of repeating numbers which contain 4    
elif(result == 4):    
    print(str(num) + " is not a happy number after apply way 1");  





#way 2:

#Another way to do this and code is also less
n=num
setData=set()		#set datastructure for checking a number is repeated or not.
while 1:
	if n==1:
		print("{} is a happy number after apply way 2".format(num))
		break
	if n in setData:
		print("{} is Not a happy number after apply way 2".format(num))
		break
	else:
		setData.add(n)	#adding into set if not inside set
		n=int(''.join(str(sum([int(i)**2 for i in str(n)]))))       #Pythonic way
