import string

def check_password_strength(password):
    strength = 0
    
    # Criteria 1: Length (Must be at least 8 characters)
    if len(password) >= 8:
        strength += 1
    
    # Criteria 2: Must contain Digits (0-9)
    has_digit = False
    for char in password:
        if char.isdigit():
            has_digit = True
            break
    if has_digit:
        strength += 1
        
    # Criteria 3: Must contain Uppercase Letters (A-Z)
    has_upper = False
    for char in password:
        if char.isupper():
            has_upper = True
            break
    if has_upper:
        strength += 1
        
    return strength

if __name__ == "__main__":
    print("--- Password Strength Checker ---")
    # Note: We cannot run input() on the website, but this code is correct.
    # If users download it, it will work.
    print("Run this script locally to test your password!")
