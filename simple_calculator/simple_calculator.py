# Define functions for each operation
def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    # Prevent division by zero
    if y == 0:
        return "Error: Division by zero is not allowed"
    return x / y

# Display the options to the user
def display_menu():
    print("\nSelect operation:")
    print("1. Add")
    print("2. Subtract")
    print("3. Multiply")
    print("4. Divide")

# Main program loop
while True:
    display_menu()
    
    # Take input from the user
    choice = input("Enter choice (1/2/3/4): ")

    # Check if the choice is valid
    if choice in ('1', '2', '3', '4'):
        try:
            num1 = float(input("Enter first number: "))
            num2 = float(input("Enter second number: "))
        except ValueError:
            print("Invalid input. Please enter numeric values.")
            continue

        # Perform the chosen operation
        if choice == '1':
            print(f"{num1} + {num2} = {add(num1, num2)}")
        elif choice == '2':
            print(f"{num1} - {num2} = {subtract(num1, num2)}")
        elif choice == '3':
            print(f"{num1} * {num2} = {multiply(num1, num2)}")
        elif choice == '4':
            print(f"{num1} / {num2} = {divide(num1, num2)}")

        # Check if the user wants another calculation
        next_calculation = input("Do you want to perform another calculation? (yes/no): ").lower()
        if next_calculation != 'yes':
            print("Exiting the calculator.")
            break
    else:
        print("Invalid input. Please select a valid operation.")
