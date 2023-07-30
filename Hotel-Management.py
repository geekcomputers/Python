import json
import sys

def menu():
    options = {
        1: {"title": "Add new customer details", "method": add},
        2: {"title": "Modify already existing customer details", "method": modify},
        3: {"title": "Search customer details", "method": search},
        4: {"title": "View all customer details", "method": view},
        5: {"title": "Delete customer details", "method": remove},
        6: {"title": "Exit the program", "method": sys.exit},
    }

    print(f"\n\n{' '*25}Welcome to Hotel Database Management Software\n\n")

    for num, option in options.items():
        print(f"{num}: {option.get('title')}")
    print()

    options.get(int(input("Enter your choice(1-6): "))).get("method")()


def add():
    Name1 = input("\nEnter your first name: ")
    Name2 = input("\nEnter your last name: ")
    Phone_Num = input("\nEnter your phone number (without +91): ")

    print("These are the rooms that are currently available")
    print("1-Normal (500/Day)")
    print("2-Deluxe (1000/Day)")
    print("3-Super Deluxe (1500/Day)")
    print("4-Premium Deluxe (2000/Day)")

    Room_Type = int(input("\nWhich type you want (1-4): "))

    room_types = {
        1: ("Normal", 500),
        2: ("Deluxe", 1000),
        3: ("Super Deluxe", 1500),
        4: ("Premium Deluxe", 2000)
    }

    Room_Type, x = room_types.get(Room_Type, ("Unknown", 0))

    Days = int(input("How many days will you stay: "))
    Money = x * Days

    print(f"\nYou have to pay {Money:.2f}")
    Payment = input("Mode of payment (Card/Cash/Online): ").capitalize()
    print()

    with open("Management.txt", "r") as file:
        dictionary = json.load(file)

    if not dictionary["Room"]:
        Room_num = "501"
    else:
        Room_num = str(int(dictionary["Room"][-1]) + 1)

    print(f"You have been assigned Room Number {Room_num}")
    print(f"Name: {Name1} {Name2}")
    print(f"Phone number: +91{Phone_Num}")
    print(f"Room type: {Room_Type}")
    print(f"Stay (day): {Days}")

    dictionary["First_Name"].append(Name1)
    dictionary["Last_Name"].append(Name2)
    dictionary["Phone_num"].append(Phone_Num)
    dictionary["Room_Type"].append(Room_Type)
    dictionary["Days"].append(Days)
    dictionary["Price"].append(Money)
    dictionary["Room"].append(Room_num)

    with open("Management.txt", "w", encoding="utf-8") as file:
        json.dump(dictionary, file)

    print("\nYour data has been successfully added to our database.")
    exit_menu()


def modify():
    with open("Management.txt", "r") as file:
        dictionary = json.load(file)

    dict_num = dictionary["Room"]
    dict_len = len(dict_num)
    if dict_len == 0:
        print("\nThere is no data in our database\n")
        menu()
    else:
        Room = input("\nEnter your Room Number: ")

        listt_num = dictionary["Room"]
        index = int(listt_num.index(Room))

        print("\n1-Change your first name")
        print("2-Change your last name")
        print("3-Change your phone number")

        choice = int(input("\nEnter your choice: "))
        print()

        with open("Management.txt", "w", encoding="utf-8") as file:
            if choice == 1:
                category = "First_Name"
            elif choice == 2:
                category = "Last_Name"
            elif choice == 3:
                category = "Phone_num"

            user_input = input(f"Enter New {category.replace('_', ' ')}: ")
            dictionary[category][index] = user_input

            json.dump(dictionary, file)

        print("\nYour data has been successfully updated")
        exit_menu()


def search():
    with open("Management.txt") as file:
        dictionary = json.load(file)

    dict_num = dictionary["Room"]
    dict_len = len(dict_num)

    if dict_len == 0:
        print("\nThere is no data in our database\n")
        menu()
    else:
        Room = input("\nEnter your Room Number: ")

        listt_num = dictionary["Room"]
        index = int(listt_num.index(Room))

        print(f"\nFirst Name: {dictionary['First_Name'][index]}")
        print(f"Last Name: {dictionary['Last_Name'][index]}")
        print(f"Phone number: {dictionary['Phone_num'][index]}")
        print(f"Room Type: {dictionary['Room_Type'][index]}")
        print(f"Days staying: {dictionary['Days'][index]}")
        print(f"Money paid: {dictionary['Price'][index]}")
        print(f"Room Number: {dictionary['Room'][index]}")

        exit_menu()


def remove():
    with open("Management.txt") as file:
        dictionary = json.load(file)

    dict_num = dictionary["Room"]
    dict_len = len(dict_num)
    if dict_len == 0:
        print("\nThere is no data in our database\n")
        menu()
    else:
        Room = input("\nEnter your Room Number: ")

        listt = dictionary["Room"]
        index = int(listt.index(Room))

        del dictionary["First_Name"][index]
        del dictionary["Last_Name"][index]
        del dictionary["Phone_num"][index]
        del dictionary["Room_Type"][index]
        del dictionary["Days"][index]
        del dictionary["Price"][index]
        del dictionary["Room"][index]

        with open("Management.txt", "w", encoding="utf-8") as file:
            json.dump(dictionary, file)

        print("Details have been removed successfully")

        exit_menu()


def view():
    with open("Management.txt") as file:
        dictionary = json.load(file)

    dict_num = dictionary["Room"]
    dict_len = len(dict_num)
    if dict_len == 0:
        print("\nThere is no data in our database\n")
        menu()
    else:
        listt = dictionary["Room"]
        a = len(listt)

        index = 0
        while index != a:
            print("")
            print("First Name:", dictionary["First_Name"][index])
            print("Last Name:", dictionary["Last_Name"][index])
            print("Phone number:", dictionary["Phone_num"][index])
            print("Room Type:", dictionary["Room_Type"][index])
            print("Days staying:", dictionary["Days"][index])
            print("Money paid:", dictionary["Price"][index])
            print("Room Number:", dictionary["Room"][index])
            print("")

            index = index + 1

        exit_menu()


def exit_menu():
    print("")
    print("Do you want to exit the program or return to the main menu")
    print("1-Main Menu")
    print("2-Exit")
    print("")

    user_input = int(input("Enter your choice: "))
    if user_input == 2:
        sys.exit()
    elif user_input == 1:
        menu()


try:
    menu()
except KeyboardInterrupt:
    print("\nexiting...!")
