# Lets say we want to print a combination of stars as shown below.

# *
# * *
# * * *
# * * * *
# * * * * *

for i in range(1,6):
    for j in range(0,i):
        print('*',end = " ")

    for j in range(1,(2*(5-i))+1):
        print(" ",end = "")

    print("")


# Let's say we want to print pattern which is opposite of above:
#  * * * * *
#    * * * *
#      * * *
#        * *
#          *

print(" ")

for i in range(1,6):

    for j in range(0,(2*(i-1))+1):
        print(" ", end="")


    for j in range(0,6-i):
        print('*',end = " ")

    print("")