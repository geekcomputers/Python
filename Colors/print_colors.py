import sys
from typing import NoReturn


class colors:
    """ANSI color codes for terminal output"""

    CYAN: str = "\033[36m"
    GREEN: str = "\033[32m"
    YELLOW: str = "\033[33m"
    BLUE: str = "\033[34m"
    RED: str = "\033[31m"
    ENDC: str = "\033[0m"  # Reset color to default


def printc(color: str, message: str) -> None:
    """Print a message with specified ANSI color"""
    print(f"{color}{message}{colors.ENDC}")


def exit_with_error(message: str) -> NoReturn:
    """Print error message and exit with status code 1"""
    printc(colors.RED, f"Error: {message}")
    sys.exit(1)


def main() -> None:
    """Main function with default message support"""
    # Set default message if no command-line argument is provided
    default_message: str = "Hello, Colorful World!"

    try:
        message: str = sys.argv[1] if len(sys.argv) > 1 else default_message
    except IndexError:
        exit_with_error("Invalid command-line arguments")

    # Print usage hint (optional, helps users understand)
    if len(sys.argv) <= 1:
        print(
            "Using default message (provide an argument to customize: python print_colors.py 'your text')\n"
        )

    # Print message in different colors
    printc(colors.CYAN, message)
    printc(colors.GREEN, message)
    printc(colors.YELLOW, message)
    printc(colors.BLUE, message)
    printc(colors.RED, message)


if __name__ == "__main__":
    main()
