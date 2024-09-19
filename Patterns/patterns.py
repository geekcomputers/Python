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
    for i in range(1,lines+1): 
        print("* "*i) 
    print() 
    for i in range(lines):
            print("  "*i,"* "*(lines-i))

if __name__ == "__main__":
    main()    
