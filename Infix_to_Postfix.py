# Python program to convert infix expression to postfix 

# Class to convert the expression 
class Conversion:

    # Constructor to initialize the class variables
    def __init__(self, capacity):
        self.top = -1
        self.capacity = capacity
        # This array is used a stack
        self.array = []
        # Precedence setting
        self.output = []
        self.precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}

    # check if the stack is empty
    def isEmpty(self):
        return True if self.top == -1 else False

    # Return the value of the top of the stack
    def peek(self):
        return self.array[-1]

    # Pop the element from the stack
    def pop(self):
        if not self.isEmpty():
            self.top -= 1
            return self.array.pop()
        else:
            return "$"

    # Push the element to the stack
    def push(self, op):
        self.top += 1
        self.array.append(op)

    # A utility function to check is the given character
    # is operand
    def isOperand(self, ch):
        return ch.isalpha()

    # Check if the precedence of operator is strictly
    # less than top of stack or not
    def notGreater(self, i):
        """
        Returns True if the precedence of i is less than or equal to that of the top element on the stack.
        """
        try:
            a = self.precedence[i]
            b = self.precedence[self.peek()]
            return True if a <= b else False
        except KeyError:
            return False

    # The main function that converts given infix expression
    # to postfix expression
    def infixToPostfix(self, exp):
 """
 This function takes an infix expression and converts it to postfix.
 It uses a stack to store the operators and operands.
 The precedence of the
 operators is given as follows:

     1) Parentheses have highest precedence followed by exponentiation, then multiplication/division,
 addition/subtraction from left to right in decreasing order of precedence.
     2) Operators with same precedences are evaluated from left-to-right
 (except for exponentiation).  For example, `2 + 3 * 4` = 20 because multiplication has higher precedence than addition.  The expression `(2 + 3) * 4`
 = 20 because parentheses have higher precendence than both addition and multiplication so evaluate first before evaluating other parts of the
 expression; that is, `(2+3)*4`.

     :param exp: A string containing an infix arithmetic expression where tokens are space separated (e.g., "1 + 2").
 Tokens can be either numbers or strings representing mathematical operations (+ - / *)^ . Parentheses may also appear in the input but they do not
 affect how this function operates on any individual token within exp; i.e., this function will always evaluate expressions according to operator
 precedences regardless of whether
 """

        # Iterate over the expression for conversion
        for i in exp:
            # If the character is an operand,
            # add it to output
            if self.isOperand(i):
                self.output.append(i)

            # If the character is an '(', push it to stack
            elif i == '(':
                self.push(i)

            # If the scanned character is an ')', pop and
            # output from the stack until and '(' is found
            elif i == ')':
                while ((not self.isEmpty()) and self.peek() != '('):
                    a = self.pop()
                    self.output.append(a)
                if (not self.isEmpty() and self.peek() != '('):
                    return -1
                else:
                    self.pop()

                # An operator is encountered
            else:
                while (not self.isEmpty() and self.notGreater(i)):
                    self.output.append(self.pop())
                self.push(i)

            # pop all the operator from the stack
        while not self.isEmpty():
            self.output.append(self.pop())

        print("".join(self.output))


# Driver program to test above function
exp = "a+b*(c^d-e)^(f+g*h)-i"
obj = Conversion(len(exp))
obj.infixToPostfix(exp)
