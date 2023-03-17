# Lets say we want to print a combination of stars as shown below.

# *
# * *
# * * *
# * * * *
# * * * * *


# Let's say we want to print pattern which is opposite of above:
#  * * * * *
#    * * * *
#      * * *
#        * *
#          *

def main():
    lines = int(input("Enter no.of lines: "))
    pattern(lines)

def pattern(lines):
    for i in range(lines):
        for j in range(i+1):
            print("* ", end="")
        print("")
    print(" ")

    for i in range(0,lines):
    
        for j in range(0, (2 * (i - 1)) + 1):
            print(" ", end="")
    
        for j in range(0, lines - i):
            print("*", end=" ")
    
        print("")    

if __name__ == "__main__":
    main()    
