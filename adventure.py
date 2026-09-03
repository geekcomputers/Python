name = input("Enter your name: ")
print(f"Welcome {name} to this adventure!")

answer = input(
    "You are on dirt road it has come to an end and you can go left or right.which way you would like to go? ").lower()
 
# """
# The results you will face when you decide to take left.
# """

if answer == "left":
    answer = input("You come to a river, you can walk around it or swim across? Type to walk around and swim to swim across: ").lower()
    if answer == "swim":
        print("You swam and got eaten by an alligator.")
    elif answer == "walk": 
        print("You walked for many miles and ran out of water ultmately made you lose.")
    else:               
        print("Not a valid option.")
 
# """
# The results you will face when you decide to take left.
# """      
 
elif answer == "right":
    answer = input("You came across the bridge it looks so old and dangerous to cross, if you wanna cross it say cross  or for head back say head back:  ").lower()
    if answer == "cross":
        print("You succesfully crossed the bridge.")
    elif answer == "head back":
        print("You lost the road and got lost. this lead to your end.")
    else:
        print("Not a valid option.")
        
    
else:
    print("Not a valid option.")