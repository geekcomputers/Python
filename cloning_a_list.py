# Python program to copy or clone a list 
# Using the Slice Operator 
def Cloning(li1): 
    return li1[:]
  
# Driver Code 
li1 = [
    4, 
    8, 
    2, 
    10, 
    15, 
    18
] 
li2 = Cloning(li1) 
print("Original List:", li1) 
print("After Cloning:", li2) 
