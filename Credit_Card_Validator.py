# luhn algorithm


class CreditCard:
    def __init__(self, card_no):
        self.card_no = card_no

    @property
    def company(self):
        """
        Returns the company name of a credit card.

        :param self: The object
        :type self: CreditCard
        :returns: Company name of a credit card.
        """
        comp = None
        if str(self.card_no).startswith('4'):
            comp = 'Visa Card'
        elif str(self.card_no).startswith(('50', '67', '58', '63',)):
            comp = 'Maestro Card'
        elif str(self.card_no).startswith('5'):
            comp = 'Master Card'
        elif str(self.card_no).startswith('37'):
            comp = 'American Express Card'
        elif str(self.card_no).startswith('62'):
            comp = 'Unionpay Card'
        elif str(self.card_no).startswith('6'):
            comp = 'Discover Card'
        elif str(self.card_no).startswith('35'):
            comp = 'JCB Card'
        elif str(self.card_no).startswith('7'):
            comp = 'Gasoline Card'

        return 'Company : ' + comp

    def first_check(self):
        """
        :param card_no:
            A string of characters representing a credit/debit card number.
        :returns message:
            A string stating whether the input is valid
        or not.

            If the length of the input is 13 to 19 characters long, it returns "First check : Valid in terms of length."

            Else, it returns "First
        check : Check Card number once again it must be of 13 or 16 digits long."
        """
        if 13 <= len(self.card_no) <= 19:
            message = "First check : Valid in terms of length."

        else:
            message = "First check : Check Card number once again it must be of 13 or 16 digits long."
        return message

    def validate(self):
        """
        This function takes a credit card number as input and returns whether it is valid or not.
        The code first reverses the credit card number, then
        iterates through each digit in the reversed list.
        If the index of that digit is odd, then we double it and sum all digits if there are two digits
        after doubling (e.g., 4 becomes 8 which is 2 + 6). 
        Otherwise, we just add that digit to our sum_ variable. We return 'Valid Card' if sum_ % 10 == 0
        else 'Invalid Card'.
        """
        # double every second digit from right to left
        sum_ = 0
        crd_no = self.card_no[::-1]
        for i in range(len(crd_no)):
            if i % 2 == 1:
                double_it = int(crd_no[i]) * 2

                if len(str(double_it)) == 2:
                    sum_ += sum([eval(i) for i in str(double_it)])

                else:
                    sum_ += double_it

            else:
                sum_ += int(crd_no[i])

        if sum_ % 10 == 0:
            response = "Valid Card"
        else:
            response = 'Invalid Card'

        return response

    @property
    def checksum(self):
        return '#CHECKSUM# : ' + self.card_no[-1]

    @classmethod
    def set_card(cls, card_to_check):
        return cls(card_to_check)


card_number = input()
card = CreditCard.set_card(card_number)
print(card.company)
print('Card : ', card.card_no)
print(card.first_check())
print(card.checksum)
print(card.validate())

# 79927398713
# 4388576018402626
# 379354508162306
