"""
Class resposible for counting words for different files:
- Reduce redundant code
- Easier code management/debugging
- Code readability
"""

## ! Is there any other way than doing it linear?


## ! What will be test cases of it?
# ! Please do let me know.
## ! Can add is digit, isspace methods too later on.
# ! Based on requirements of it



## ! The questions are nothing but test-cases
## ! Make a test thing and handle it.
# does it count only alphabets or numerics too?
# ? what about other characters?
class Counter:
    def __init__(self, text: str) -> None:
        self.text = text
        # Define the initial count of the lower and upper case.
        self.count_lower = 0
        self.count_upper = 0
        self.compute()

    def compute(self) -> None:
        for char in self.text:
            if char.islower():
                self.count_lower += 1
            elif char.isupper():
                self.count_upper += 1

    def get_total_lower(self) -> int:
        return self.count_lower

    def get_total_upper(self) -> int:
        return self.count_upper

    def get_total_chars(self) -> int:
        return self.count_lower + self.count_upper
