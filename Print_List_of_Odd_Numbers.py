# INPUT NUMBER OF ODD NUMBERS
def print_error_messages():
  print("Invalid number, please enter a Non-negative number!")
  exit()


try:
  n=int(input('Amount: '))
except ValueError:
  print_error_messages()

start=1

if n < 0:
  print_error_messages()

for i in range(n):
  print(start)
  start+=2
