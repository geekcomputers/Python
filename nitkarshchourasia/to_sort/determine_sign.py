from word2number import w2n

class DetermineSign:
    def __init__(self, num: str | float | None = None) -> None:
        """
        Initialize with a number or prompt user input.
        
        Args:
            num: A number (float/int) or string representation (e.g., "5" or "five").
        """
        if num is None:
            self.num = self._get_valid_number()
        else:
            self.num = self._convert_to_float(num)
        self.num = round(self.num, 1)  # Single rounding at initialization

    def _convert_to_float(self, input_value: str | float) -> float:
        """
        Convert input to float, handling both numeric strings and word representations.
        
        Args:
            input_value: A string (e.g., "5", "five") or numeric value.
            
        Returns:
            The float representation.
            
        Raises:
            ValueError: If input cannot be converted to a valid number.
        """
        if isinstance(input_value, (int, float)):
            return float(input_value)
        
        try:
            # Handle numeric strings (e.g., "5.5")
            return float(input_value)
        except ValueError:
            try:
                # Handle word representations (e.g., "five point five")
                # Replace "point" with decimal separator for word2number
                if "point" in input_value.lower():
                    parts = input_value.lower().split("point")
                    integer_part = w2n.word_to_num(parts[0].strip())
                    decimal_part = w2n.word_to_num(parts[1].strip())
                    return float(f"{integer_part}.{decimal_part}")
                else:
                    return float(w2n.word_to_num(input_value))
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid input: '{input_value}'. Please enter a valid number or word."
                )

    def _get_valid_number(self) -> float:
        """
        Prompt user for input until a valid number is provided.
        
        Returns:
            A validated float.
        """
        while True:
            user_input = input("Enter a number: ").strip()
            try:
                return self._convert_to_float(user_input)
            except ValueError as e:
                print(f"Error: {e}. Try again.")

    def determine_sign(self) -> str:
        """Determine if the number is positive, negative, or zero."""
        if self.num > 0:
            return "Positive number"
        elif self.num < 0:
            return "Negative number"
        else:
            return "Zero"

    def __repr__(self) -> str:
        return f"Number: {self.num}, Sign: {self.determine_sign()}"


if __name__ == "__main__":
    number1 = DetermineSign()
    print(f"The number is {number1.determine_sign()}")