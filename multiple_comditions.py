while True:
    try:
        user = int(input("enter any number b/w 1-3\n"))
        if user == 1:
            print("in first if")
        elif user == 2:
            print("in second if")
        elif user == 3:
            print("in third if")
        else:
            print("Enter numbers b/w the range of 1-3")
    except:
        print("enter only digits")


"""
## Why we are using elif instead of nested if ?
When you have multiple conditions to check, using nested if means that if the first condition is true, the program still checks the second 
if condition, even though it's already decided that the first condition worked. This makes the program do more work than necessary.
On the other hand, when you use elif, if one condition is satisfied, the program exits the rest of the conditions and doesn't continue checking. 
It’s more efficient and clean, as it immediately moves to the correct option without unnecessary steps.
"""
