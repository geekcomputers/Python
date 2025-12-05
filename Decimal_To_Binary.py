"""
Decimal to Binary Converter

This program converts base-10 decimal numbers (including fractional values) 
to their binary representation. The conversion handles both integer and 
fractional parts with configurable precision.

Features:
- Converts both integer and fractional decimal numbers to binary
- Configurable precision for fractional part conversion
- Handles negative numbers
- Type annotations for better code clarity
- Robust error handling for various input scenarios
"""

decimal_accuracy = 7  # Precision for fractional part conversion


def decimal_to_binary(number: float) -> str:
    """
    Convert a decimal number to its binary representation.
    
    Args:
        number (float): The base-10 number to convert to binary
        
    Returns:
        str: Binary representation of the input number
    """
    # Handle special cases
    if number == 0:
        return "0"
    
    # Handle negative numbers
    is_negative = number < 0
    number = abs(number)
    
    # Separate integer and fractional parts
    integer_part = int(number)
    fractional_part = round(number - integer_part, decimal_accuracy)
    
    # Convert integer part to binary
    integer_binary = _convert_integer_part(integer_part)
    
    # Convert fractional part to binary
    fractional_binary = _convert_fractional_part(fractional_part)
    
    # Combine parts and add sign if needed
    result = integer_binary + fractional_binary
    return f"-{result}" if is_negative else result


def _convert_integer_part(integer: int) -> str:
    """
    Convert integer part to binary using division method.
    
    Args:
        integer (int): Integer part of the number
        
    Returns:
        str: Binary representation of the integer part
    """
    if integer == 0:
        return "0"
    
    binary_digits = []
    num = integer
    
    while num > 0:
        binary_digits.append(str(num % 2))  # Get remainder
        num = num // 2  # Integer division
    
    # Reverse the list to get correct binary order
    binary_digits.reverse()
    return "".join(binary_digits)


def _convert_fractional_part(fraction: float) -> str:
    """
    Convert fractional part to binary using multiplication method.
    
    Args:
        fraction (float): Fractional part of the number (0 <= fraction < 1)
        
    Returns:
        str: Binary representation of the fractional part
    """
    if fraction == 0:
        return ""
    
    binary_digits = ["."]
    current_fraction = fraction
    iterations = 0
    
    # Convert fractional part until it becomes 0 or reaches maximum precision
    while current_fraction > 0 and iterations < decimal_accuracy:
        # Multiply by 2 and take integer part
        current_fraction *= 2
        integer_part = int(current_fraction)
        binary_digits.append(str(integer_part))
        
        # Keep only the fractional part for next iteration
        current_fraction -= integer_part
        current_fraction = round(current_fraction, decimal_accuracy)
        iterations += 1
    
    return "".join(binary_digits)


def decimal_to_binary_recursive(number: int) -> str:
    """
    Alternative recursive implementation for integer conversion only.
    
    Note: This version only works with integers and doesn't handle fractional parts.
    
    Args:
        number (int): Integer number to convert to binary
        
    Returns:
        str: Binary representation of the integer
    """
    if number > 1:
        return decimal_to_binary_recursive(number // 2) + str(number % 2)
    return str(number)


def get_user_input() -> float:
    """
    Safely get user input with comprehensive error handling.
    
    Returns:
        float: Validated decimal number from user input
        
    Raises:
        KeyboardInterrupt: If user interrupts the program
        EOFError: If no input is available (non-interactive environment)
    """
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            user_input = input("Enter any base-10 number (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                exit(0)
                
            return float(user_input)
            
        except ValueError:
            print(f"Error: '{user_input}' is not a valid number. Please try again.")
            if attempt < max_attempts - 1:
                print(f"{max_attempts - attempt - 1} attempts remaining.")
            else:
                print("Maximum attempts reached. Using default value 0.")
                return 0.0
        except (EOFError, KeyboardInterrupt):
            raise


def main() -> None:
    """Main function to run the decimal to binary converter."""
    print("=== Decimal to Binary Converter ===")
    print(f"Fractional precision: {decimal_accuracy} bits")
    print("Enter 'quit' to exit the program")
    print("-" * 40)
    
    try:
        while True:
            try:
                # Get input from user
                number = get_user_input()
                
                # Convert to binary using main method
                binary_result = decimal_to_binary(number)
                
                # Display results
                print(f"\nDecimal number: {number}")
                print(f"Binary representation: {binary_result}")
                
                # For integer inputs, show recursive method as well
                if number.is_integer() and number >= 0:
                    recursive_result = decimal_to_binary_recursive(int(number))
                    print(f"Recursive method (integer only): {recursive_result}")
                
                print("-" * 40)
                
            except (KeyboardInterrupt, EOFError):
                print("\n\nProgram terminated by user. Goodbye!")
                break
            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")
                print("Please try again with a different number.")
                print("-" * 40)
                
    except (KeyboardInterrupt, EOFError):
        print("\n\nProgram terminated. Goodbye!")


def run_example() -> None:
    """
    Run example conversions for demonstration purposes.
    Useful when running in non-interactive environments.
    """
    print("=== Example Conversions (Non-interactive Mode) ===")
    examples = [10.0, 15.75, -3.125, 0.5, 255.255]
    
    for example in examples:
        binary_result = decimal_to_binary(example)
        print(f"Decimal: {example:8} -> Binary: {binary_result}")
    
    print("\nTo use interactive mode, run the program in a terminal.")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\n\nNo interactive input available. Running in example mode.")
        run_example()
