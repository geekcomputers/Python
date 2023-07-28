"""
    Class resposible for counting words for different files:
    - Reduce redundant code
    - Easier code management/debugging
    - Code readability
"""

class Counter:

    def __init__(self, text:str) -> None:
        self.text = text

        # Define the initial count of the lower and upper case.
        self.count_lower = 0
        self.count_upper = 0
        self.count()

    def count(self) -> None:
        
        for char in self.text:
            if char.lower():
                self.count_lower += 1
            elif char.upper():
                self.count_upper += 1

        return (self.count_lower, self.count_upper)
    
    def get_total_lower(self) -> int:
        return self.count_lower

    def get_total_upper(self) -> int:
        return self.count_upper

    def get_total(self) -> int:
        return self.count_lower + self.count_upper