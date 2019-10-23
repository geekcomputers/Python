
# This method is called exhaustive numeration!
# I am checking every possible value
# that can be root of given x systematically
# Kinda brute forcing


def cubeRoot():
    x = int(input("Enter an integer: "))
    for ans in range(0, abs(x) + 1):
        if ans ** 3 == abs(x):
            break
    if ans ** 3 != abs(x):
        print(x, 'is not a perfect cube!')
    else:
        if x < 0:
            ans = -ans
    print('Cube root of ' + str(x) + ' is ' + str(ans))
    
cubeRoot()

cont = str(input("Would you like to continue: "))
while cont == "yes":
    cubeRoot()
    cont = str(input("Would you like to continue: "))
    if cont == "no":
        exit()
    else:
        print("Enter a correct answer(yes or no)")
        cont = str(input("Would you like to continue: "))


