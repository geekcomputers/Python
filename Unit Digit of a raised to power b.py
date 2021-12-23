def last_digit(a, b):
    """
    This function takes two integers a and b as input.
    It returns the last digit of a^b where b is an integer less than or equal to 10^7.
    The code assumes
    that 0^0 is 1, and that if the last digit of a number is in [0,5,6,1] then it will return this same digit. 
    If b%4==0 then it returns ((a%10)**4)%10
    else it returns ((a%10)**(b%4))%.
    """
    if b==0:   #This Code assumes that 0^0 is 1
        return 1
    elif a%10 in [0,5,6,1]:
        return a%10
    elif b%4==0:
        return ((a%10)**4)%10
    else:
        return ((a%10)**(b%4))%10
    
#Courtesy to https://brilliant.org/wiki/finding-the-last-digit-of-a-power/
