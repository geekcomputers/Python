def factorial(num: int):
       if num < 0:
              print("Factorial can't be calculated for numbers less than zero.")
       elif num == 0:
              print("Factorial of 0 is 1.")
       else:
              print(f"Factorial of {num} is {num*factorial(num-1)}.")
n = int(input("Enter a number to find factorial: "))
factorial(n)
