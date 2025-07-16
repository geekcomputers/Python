import sys

class colors:
    """ANSI color codes for terminal output"""
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RED = "\033[31m"
    ENDC = "\033[0m"  # Reset color to default

def printc(color: str, message: str) -> None:
    """Print a message with specified ANSI color"""
    print(f"{color}{message}{colors.ENDC}")

def main() -> None:
    """Main function with default message support"""
    # Set default message if no command-line argument is provided
    default_message = "Hello, Colorful World!"
    message = sys.argv[1] if len(sys.argv) > 1 else default_message
    
    # Print usage hint (optional, helps users understand)
    if len(sys.argv) <= 1:
        print("Using default message (provide an argument to customize: python print_colors.py 'your text')\n")
    
    # Print message in different colors
    printc(colors.CYAN, message)
    printc(colors.GREEN, message)
    printc(colors.YELLOW, message)
    printc(colors.BLUE, message)
    printc(colors.RED, message)

if __name__ == "__main__":
    main()