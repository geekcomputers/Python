# Lets say we want to print a combination of stars as shown below.

# *
# * *
# * * *
# * * * *
# * * * * *

for i in range(1, 6):
    for _ in range(i):
        print("*", end=" ")

    for _ in range(1, (2 * (5 - i)) + 1):
        print(" ", end="")

    print("")


# Let's say we want to print pattern which is opposite of above:
#  * * * * *
#    * * * *
#      * * *
#        * *
#          *

print(" ")

for i in range(1, 6):

    for _ in range((2 * (i - 1)) + 1):
        print(" ", end="")

    for _ in range(6 - i):
        print("*", end=" ")

    print("")
