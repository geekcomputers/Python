def is_armstrong_number(number: str) -> bool:
    """Check if a number (as a string) is a narcissistic/Armstrong number."""
    # Logic: Get the exponent (number of digits)
    exponent = len(number)
    
    # Logic: Sum each digit raised to the power in a single line
    # This uses a generator, which is memory efficient.
    total = sum(int(digit) ** exponent for digit in number)
    
    # Return the boolean result instead of printing
    return total == int(number)

# --- Main execution ---
user_input = input("Enter the number: ")

if is_armstrong_number(user_input):
    print(f"{user_input} is an Armstrong number")
else:
    print(f"{user_input} is not an Armstrong number")
