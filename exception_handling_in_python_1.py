# Exception handling using python


a = 12
b = 0
# a = int(input())
# b = int(input())

try:
    c = a / b
    print(c)
    # trying to print an unknown variable d
    print(d)

except ZeroDivisionError:
    print("Invalid input. Divisor cannot be zero.")

except NameError:
    print("Name of variable not defined.")


# finally statement is always executed whether or not any errors occur
a = 5
b = 0
# a = int(input())
# b = int(input())

try:
    c = a / b
    print(c)

except ZeroDivisionError:
    print("Invalid input. Divisor cannot be zero.")

finally:
    print("Hope all errors were resolved!!")


# A few other common errors
# SyntaxError

try:
    # eval is a built-in-function used in python, eval function parses the expression argument and evaluates it as a python expression.
    eval("x === x")

except SyntaxError:
    print("Please check your syntax.")


# TypeError

try:
    a = "2" + 2

except TypeError:
    print("int type cannot be added to str type.")


# ValueError

try:
    a = int("abc")

except ValueError:
    print("Enter a valid integer literal.")


# IndexError

l = [1, 2, 3, 4]

try:
    print(l[4])

except IndexError:
    print("Index of the sequence is out of range. Indexing in python starts from 0.")


# FileNotFoundError

f = open("aaa.txt", "w")  # File aaa.txt created
f.close()

try:
    # Instead of aaa.txt lets try opening abc.txt
    f = open("abc.txt", "r")

except FileNotFoundError:
    print("Incorrect file name used")

finally:
    f.close()


# Handling multiple errors in general

try:
    a = 12 / 0
    b = "2" + 2
    c = int("abc")
    eval("x===x")

except:
    pass

finally:
    print(
        "Handled multiples errors at one go with no need of knowing names of the errors."
    )


# Creating your own Error

a = 8
# a = int(input())

if a < 18:
    raise Exception("You are legally underage!!!")

else:
    print("All is well, go ahead!!")
