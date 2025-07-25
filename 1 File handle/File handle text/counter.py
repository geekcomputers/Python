class TextCounter:
    """
    A utility class for counting characters in text or files.

    Features:
    - Counts lowercase, uppercase, digits, and special characters
    - Handles both text input and file reading
    - Provides detailed character statistics

    Example usage:
    >>> counter = TextCounter("Hello World!")
    >>> counter.get_total_lower()
    8
    >>> counter.get_total_upper()
    2
    >>> counter.get_stats()
    {'lowercase': 8, 'uppercase': 2, 'digits': 0, 'special': 1, 'total': 11}
    """

    def __init__(self, content: str, is_file_path: bool = False) -> None:
        """
        Initialize the counter with text or file content.

        Args:
            content: Text string or file path
            is_file_path: Set to True if content is a file path

        Raises:
            FileNotFoundError: If the specified file does not exist
            PermissionError: If the file cannot be accessed
            UnicodeDecodeError: If the file cannot be decoded properly
        """
        self.text: str = self._read_content(content, is_file_path)
        self.count_lower: int = 0
        self.count_upper: int = 0
        self.count_digits: int = 0
        self.count_special: int = 0
        self._count_characters()

    def _read_content(self, content: str, is_file_path: bool) -> str:
        """
        Read content from file if needed, otherwise return the text directly.

        Args:
            content: Text string or file path
            is_file_path: Set to True if content is a file path

        Returns:
            The content as a string

        Raises:
            FileNotFoundError: If the file does not exist
            PermissionError: If the file cannot be accessed
            UnicodeDecodeError: If the file cannot be decoded properly
        """
        if is_file_path:
            with open(content, encoding="utf-8") as file:
                return file.read()
        return content

    def _count_characters(self) -> None:
        """
        Count different character types in the text.

        Updates the following attributes:
            count_lower: Number of lowercase letters
            count_upper: Number of uppercase letters
            count_digits: Number of digits
            count_special: Number of special characters (non-alphanumeric)
        """
        for char in self.text:
            if char.islower():
                self.count_lower += 1
            elif char.isupper():
                self.count_upper += 1
            elif char.isdigit():
                self.count_digits += 1
            else:
                self.count_special += 1

    def get_total_lower(self) -> int:
        """Return the count of lowercase characters."""
        return self.count_lower

    def get_total_upper(self) -> int:
        """Return the count of uppercase characters."""
        return self.count_upper

    def get_total_digits(self) -> int:
        """Return the count of digit characters."""
        return self.count_digits

    def get_total_special(self) -> int:
        """Return the count of special characters."""
        return self.count_special

    def get_total(self) -> int:
        """Return the total count of characters."""
        return (
            self.count_lower + self.count_upper + self.count_digits + self.count_special
        )

    def get_stats(self) -> dict[str, int]:
        """
        Return detailed character statistics.

        Returns:
            A dictionary containing counts for lowercase, uppercase,
            digits, special characters, and the total count.
        """
        return {
            "lowercase": self.count_lower,
            "uppercase": self.count_upper,
            "digits": self.count_digits,
            "special": self.count_special,
            "total": self.get_total(),
        }
