import re # importing regrex here
# Here we are tyring to validate credit/debit card number
# It must contain exactly 16 digits.
# It must start with a 4,5 or 6
# It must only consist of digits (0-9).
# It may have digits in groups of 4 , separated by one hyphen "-".
# It must NOT use any other separator like ' ' , '_', etc.

def validate_card_number(number):
    PATTERN='^([456][0-9]{3})-?([0-9]{4})-?([0-9]{4})-?([0-9]{4})$'
    result = re.match(PATTERN,number)
    if result:
        return True
    else:
        return False
if __name__=="__main__":
    # ex: 0000-1111-2222-3333 -- This result will be false
    # ex: 0000-1111-2222-3333 -- Return False
    # ex: 444-55555-6666-7777 -- Return False
    # ex: 4011-7505-1047-1848 -- Returns True
    # ex: 6015399610667820    -- Returns True
    # ex: 4444-3333-2222-XXXX -- Returns False
    print(validate_card_number(4444-3333-2222-XXXX))
    
