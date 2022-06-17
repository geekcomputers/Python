# Python program to reverse a string using stack

# Function to create an empty stack.
# It initializes size of stack as 0
def createStack():
    return []


# Function to determine the size of the stack
def size(stack):
    return len(stack)


# Stack is empty if the size is 0
def isEmpty(stack):
    if size(stack) == 0:
        return True


# Function to add an item to stack .
# It increases size by 1
def push(stack, item):
    stack.append(item)


# Function to remove an item from stack.
# It decreases size by 1
def pop(stack):
    if isEmpty(stack):
        return
    return stack.pop()


# A stack based function to reverse a string
def reverse(string):
    n = len(string)

    # Create a empty stack
    stack = createStack()

    # Push all characters of string to stack
    for i in range(n):
        push(stack, string[i])

    # Making the string empty since all
    # characters are saved in stack
    string = "".join(pop(stack) for _ in range(n))

    return string


# Driver program to test above functions
string = "GeeksQuiz"
string = reverse(string)
print(f"Reversed string is {string}")

# This code is contributed by Yash
