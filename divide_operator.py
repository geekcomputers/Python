class DivisionOperation:
    INT_MAX = float('inf')

    def __init__(self, num1, num2):
        self.num1 = num1
        self.num2 = num2

    def perform_division(self):
        if self.num1 == 0:
            return 0
        if self.num2 == 0:
            return self.INT_MAX

        neg_result = False

        # Handling negative numbers
        if self.num1 < 0:
            self.num1 = -self.num1

            if self.num2 < 0:
                self.num2 = -self.num2
            else:
                neg_result = True
        elif self.num2 < 0:
            self.num2 = -self.num2
            neg_result = True

        quotient = 0

        while self.num1 >= self.num2:
            self.num1 -= self.num2
            quotient += 1

        if neg_result:
            quotient = -quotient
        return quotient


# Driver program
num1 = 13
num2 = 2

# Create a DivisionOperation object and pass num1, num2 as arguments
division_op = DivisionOperation(num1, num2)

# Call the perform_division method of the DivisionOperation object
result = division_op.perform_division()

print(result)
