# Python program to swap two variables

x = 5
y = 10

# To take inputs from the user
#x = input('Enter value of x: ')
#y = input('Enter value of y: ')

# create a temporary variable and swap the values
temp = x
x = y
y = temp

# in python we can swap without using a third variablex

x,y = y,x
print(x,y)
# output is 10,5

print('The value of x after swapping: {}'.format(x))
print('The value of y after swapping: {}'.format(y))
