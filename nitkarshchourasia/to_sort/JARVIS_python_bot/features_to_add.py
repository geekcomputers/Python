#!/usr/bin/env python3

import time
from getpass import getuser


def get_user_name() -> str:
    """
    Ask the user for their name, using the system username as a suggestion.
    Handles edge cases and ensures a non-empty name is returned.
    """
    system_username = getuser()
    
    while True:
        name_check = input(f"Is your name {system_username}? → ").strip().lower()
        
        if name_check.startswith("y"):
            print("Okay.")
            time.sleep(0.5)  # Shorter delay for better UX
            return system_username
            
        elif name_check.startswith("n"):
            while True:
                custom_name = input("Then what is it? → ").strip()
                if custom_name:  # Ensure non-empty name
                    return custom_name
                print("Please enter a valid name.")
                
        else:
            print("Please answer with 'yes' or 'no'.")

if __name__ == "__main__":
    user_name = get_user_name()
    print(f"Nice to meet you, {user_name}!")  # Optional confirmation