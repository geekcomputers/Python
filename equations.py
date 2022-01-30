###
#####
####### by @JymPatel
#####
###

###
##### edited by ... (editors can put their name and thanks for suggestion) :)
###


# what we are going to do
print("We can solve the below equations")
print("1  Quadratic Equation")

# ask what they want to solve
sinput = input("What you would like to solve?")

# for Qdc Eqn
if sinput == "1":
    print("We will solve for equation 'a(x^2) + b(x) + c'")

    # value of a
    a = int(input("What is value of a?"))
    b = int(input("What is value of b?"))
    c = int(input("What is value of c?"))

    D = b ** 2 - 4 * a * c

    if D < 0:
        print("No real values of x satisfies your equation.")

    else:
        x1 = (-b + D) / (2 * a)
        x2 = (-b - D) / (2 * a)

        print("Roots for your equation are", x1, "&", x2)


else:
    print("You have selected wrong option.")
    print("Select integer for your equation and run this code again")


# end of code
print("You can visit https://github.com/JymPatel/Python3-FirstEdition")

# get NEW versions of equations.py at https://github.com/JymPatel/Python3-FirstEdition with more equations
# EVEN YOU CAN CONTRIBUTE THEIR. EVERYONE IS WELCOMED THERE..
