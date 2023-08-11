q=0   # Initially we assigned 0 to "q", to use this variable for the summation purpose below.
    # The "q" value should be declared before using it(mandatory). And this value can be changed later.

n=int(input("Enter Number: "))   # asking user for input
while n>0:         # Until "n" is greater than 0, execute the loop. This means that until all the digits of "n" got extracted.

   r=n%10           # Here, we are extracting each digit from "n" starting from one's place to ten's and hundred's... so on.

   q=q+r            # Each extracted number is being added to "q".

   n=n//10          # "n" value is being changed in every iteration. Dividing with 10 gives exact digits in that number, reducing one digit in every iteration from one's place.

print("Sum of digits is: "+str(q))
