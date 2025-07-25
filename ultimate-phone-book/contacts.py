"""Contact Manager Application

A simple command-line contact management system that allows users to
view, add, and remove contacts. Advanced features like sorting and
toggleable prompts require a valid key.

Data is persisted using Python's pickle module.
"""

import os
import pickle

# File paths
DATA_FILE: str = "data/pickle-main"
KEY_FILE: str = "data/pickle-key"
VALID_KEY: str = "SKD0DW99SAMXI19#DJI9"

# Contact field indices
FNAME: int = 0
LNAME: int = 1
NUMBER: int = 2
EMAIL: int = 3


# Feature flags
class Features:
    SORTING: bool = False
    TOGGLE_PROMPT: bool = False


def load_data() -> list[list[str]]:
    """Load contact data from pickle file.

    Returns:
        List of lists containing contact information.
        Each inner list represents a field (first name, last name, etc.).
    """
    try:
        with open(DATA_FILE, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError):
        return [[], [], [], []]  # Initialize empty data structure


def save_data(data: list[list[str]]) -> None:
    """Save contact data to pickle file.

    Args:
        data: Contact data to save.
    """
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "wb") as f:
        pickle.dump(data, f)


def validate_key() -> bool:
    """Check if valid key file exists.

    Returns:
        True if key is valid, False otherwise.
    """
    if os.path.isfile(KEY_FILE):
        try:
            with open(KEY_FILE, "rb") as f:
                key = pickle.load(f)
            if key == VALID_KEY:
                Features.SORTING = True
                Features.TOGGLE_PROMPT = True
                print("Key verified. All features enabled.")
                return True
        except (pickle.UnpicklingError, ValueError):
            pass
    print("Key not found or invalid. Some features are disabled.")
    return False


def display_contacts(data: list[list[str]]) -> None:
    """Display all contacts in a formatted table.

    Args:
        data: Contact data to display.
    """
    print("\n== YOUR CONTACT LIST ==")
    for i in range(len(data[FNAME])):
        print(
            f"{data[FNAME][i]} {data[LNAME][i]},  {data[NUMBER][i]}  {data[EMAIL][i]}"
        )
    print("=======================")


def add_contact(data: list[list[str]]) -> None:
    """Add a new contact to the data.

    Args:
        data: Contact data to modify.
    """
    data[FNAME].append(input("First Name: "))
    data[LNAME].append(input("Last Name: "))
    data[NUMBER].append(input("Phone Number: "))
    data[EMAIL].append(input("Email ID: "))
    print("Contact added successfully.")


def remove_contact(data: list[list[str]]) -> None:
    """Remove a contact by first name.

    Args:
        data: Contact data to modify.
    """
    print("Enter first name to delete (or 'STOP' to cancel):")
    name = input("INPUT: ").strip()

    if name.upper() == "STOP":
        return

    matches = [i for i, fn in enumerate(data[FNAME]) if fn.upper() == name.upper()]

    if not matches:
        print("No contacts found with that first name.")
    elif len(matches) == 1:
        index = matches[0]
        print("Confirm deletion:")
        print(f"{data[FNAME][index]} {data[LNAME][index]}")
        print(f"{data[NUMBER][index]}")
        print(f"{data[EMAIL][index]}")

        confirm = input("Delete this contact? (y/n): ").lower()
        if confirm == "y":
            for field in data:
                del field[index]
            print("Contact removed.")
        else:
            print("Deletion cancelled.")
    else:
        print(f"Multiple contacts found ({len(matches)}). Please refine your search.")


def sort_contacts(data: list[list[str]]) -> None:
    """Sort contacts by first name, then last name (requires valid key).

    Args:
        data: Contact data to sort.
    """
    if not Features.SORTING:
        print("Sorting requires a valid key.")
        return

    # Create list of tuples for sorting
    contacts = list(zip(*data))
    contacts.sort(key=lambda x: (x[FNAME].upper(), x[LNAME].upper()))

    # Update original data structure
    for i, field in enumerate(data):
        field.clear()
        field.extend([contact[i] for contact in contacts])

    print("Contacts sorted alphabetically.")


def main() -> None:
    """Main application entry point."""
    print("Contact Manager (GPL v3 License)")
    print("https://github.com/JymPatel/Python-FirstEdition\n")

    # Load data and validate key
    data = load_data()
    has_key = validate_key()

    print("Update 22.02: Data is automatically saved when exiting.\n")

    prompt_enabled = True
    loop = True

    while loop:
        if prompt_enabled:
            print("\n0. Exit program")
            print("1. View all contacts")
            print("2. Add new contact")
            print("3. Remove contact")
            if Features.SORTING:
                print("4. Sort contacts alphabetically")
            if Features.TOGGLE_PROMPT:
                print("9. Toggle this prompt")

        try:
            choice = input("\nEnter choice: ").strip()
            if not choice:
                continue
            choice = int(choice)
        except ValueError:
            print("Please enter a valid integer.")
            continue

        # Process user choice
        if choice == 0:
            save_data(data)
            print("Data saved. Exiting...")
            loop = False
        elif choice == 1:
            display_contacts(data)
        elif choice == 2:
            add_contact(data)
        elif choice == 3:
            remove_contact(data)
        elif choice == 4 and Features.SORTING:
            sort_contacts(data)
        elif choice == 9 and Features.TOGGLE_PROMPT:
            prompt_enabled = not prompt_enabled
            status = "enabled" if prompt_enabled else "disabled"
            print(f"Prompt {status}. Press 9 again to toggle.")
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
