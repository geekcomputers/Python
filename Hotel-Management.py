
def menu():

    options = {
        1 : {
            "title" : "Add new customer details", 
            "method": lambda : add()
            },

        2 : {
            "title" : "Modify already existing customer details", 
            "method": lambda : modify()
            },

        3 : {
            "title" : "Search customer details", 
            "method": lambda : search()
            },

        4 : {
            "title" : "View all customer details", 
            "method": lambda : view()
            },

        5 : {
            "title" : "Delete customer details", 
            "method": lambda : remove()
            },

        6 : {
            "title" : "Exit the program", 
            "method": lambda : exit()
            }
    }

    print(f"\n\n{' '*25}Welcome to Hotel Database Management Software\n\n")

    for num, option in options.items():
        print(f"{num}: {option.get('title')}")
    print()

    options.get( int(input("Enter your choice(1-6): ")) ).get("method")()


def add():

    Name1 = input("\nEnter your first name: \n")
    Name2 = input("\nEnter your last name: \n")
    Phone_Num = input("\nEnter your phone number(without +91): \n")

    print("These are the rooms that are currently available")
    print("1-Normal (500/Day)")
    print("2-Deluxe (1000/Day)")
    print("3-Super Deluxe (1500/Day)")
    print("4-Premium Deluxe (2000/Day)")

    Room_Type = int(input("\nWhich type you want(1-4): \n"))

    match Room_Type:
        case 1:
            x = 500
            Room_Type = "Normal"
        case 2:
            x = 1000
            Room_Type = "Deluxe"
        case 3:
            x = 1500
            Room_Type = "Super Deluxe"
        case 4:
            x = 2000
            Room_Type = "Premium"

    Days = int(input("How many days you will stay: "))
    Money = x * Days
    Money = str(Money)
    print("")

    print("You have to pay ", (Money))
    print("")

    Payment = input("Mode of payment(Card/Cash/Online): ").capitalize()
    if Payment == "Card":
        print("Payment with card")
    elif Payment == "Cash":
        print("Payment with cash")
    elif Payment == "Online":
        print("Online payment")
    print("")

    with open("Management.txt", "r") as File:
        string = File.read()
        string = string.replace("'", '"')
        dictionary = json.loads(string)

    if len(dictionary.get("Room")) == 0:
        Room_num = "501"
    else:
        listt = dictionary.get("Room")
        tempp = len(listt) - 1
        temppp = int(listt[tempp])
        Room_num = 1 + temppp
        Room_num = str(Room_num)

    print("You have been assigned Room Number", Room_num)
    print(f"name : {Name1} {Name2}")
    print(f"phone number : +91{Phone_Num}")
    print(f"Room type : {Room_Type}")
    print(f"Stay (day) : {Days}")

    dictionary["First_Name"].append(Name1)
    dictionary["Last_Name"].append(Name2)
    dictionary["Phone_num"].append(Phone_Num)
    dictionary["Room_Type"].append(Room_Type)
    dictionary["Days"].append(Days)
    dictionary["Price"].append(Money)
    dictionary["Room"].append(Room_num)

    with open("Management.txt", "w", encoding="utf-8") as File:
        File.write(str(dictionary))

    print("\nYour data has been successfully added to our database.")

    exit_menu()


import os
import json

filecheck = os.path.isfile("Management.txt")
if not filecheck:
    with open("Management.txt", "a", encoding="utf-8") as File:
        temp1 = {
            "First_Name": [],
            "Last_Name": [],
            "Phone_num": [],
            "Room_Type": [],
            "Days": [],
            "Price": [],
            "Room": [],
        }
        File.write(str(temp1))


def modify():

    with open("Management.txt", "r") as File:
        string = File.read()
        string = string.replace("'", '"')
        dictionary = json.loads(string)

    dict_num = dictionary.get("Room")
    dict_len = len(dict_num)
    if dict_len == 0:
        print("\nThere is no data in our database\n")
        menu()
    else:
        Room = input("\nEnter your Room Number: ")

        listt = dictionary["Room"]
        index = int(listt.index(Room))

        print("\n1-Change your first name")
        print("2-Change your last name")
        print("3-Change your phone number")

        choice = int(input("\nEnter your choice: "))
        print()

        with open("Management.txt", "w", encoding="utf-8") as File:
            
            match choice:
                case 1:
                    category = "First_Name"
                case 2:
                    category = "Last_Name"
                case 3:
                    category = "Phone_num"

            user_input = input(f"Enter New {category.replace('_', ' ')}")
            listt1 = dictionary[category]
            listt1[index] = user_input
            dictionary[category] = None
            dictionary[category] = listt1

            File.write(str(dictionary))

        print("\nYour data has been successfully updated")
        exit_menu()


def search():

    with open("Management.txt") as File:
        dictionary = json.loads(File.read().replace("'", '"'))

    dict_num = dictionary.get("Room")
    dict_len = len(dict_num)

    if dict_len == 0:
        print("\nThere is no data in our database\n")
        menu()
    else:
        Room = input("\nEnter your Room Number: ")

        listt_num = dictionary.get("Room")
        index = int(listt_num.index(Room))

        listt_fname = dictionary.get("First_Name")
        listt_lname = dictionary.get("Last_Name")
        listt_phone = dictionary.get("Phone_num")
        listt_type = dictionary.get("Room_Type")
        listt_days = dictionary.get("Days")
        listt_price = dictionary.get("Price")

        print(f"\nFirst Name: {listt_fname[index]}")
        print(f"Last Name: {listt_lname[index]}")
        print(f"Phone number: {listt_phone[index]}")
        print(f"Room Type: {listt_type[index]}")
        print(f"Days staying: {listt_days[index]}")
        print(f"Money paid: {listt_price[index]}")
        print(f"Room Number: {listt_num[index]}")

        exit_menu()


def remove():
    with open("Management.txt") as File:
        dictionary = json.loads(File.read().replace("'", '"'))

    dict_num = dictionary.get("Room")
    dict_len = len(dict_num)
    if dict_len == 0:
        print("\nThere is no data in our database\n")
        menu()
    else:
        Room = input("\nEnter your Room Number: ")

        listt = dictionary["Room"]
        index = int(listt.index(Room))

        listt_fname = dictionary.get("First_Name")
        listt_lname = dictionary.get("Last_Name")
        listt_phone = dictionary.get("Phone_num")
        listt_type = dictionary.get("Room_Type")
        listt_days = dictionary.get("Days")
        listt_price = dictionary.get("Price")
        listt_num = dictionary.get("Room")

        del listt_fname[index]
        del listt_lname[index]
        del listt_phone[index]
        del listt_type[index]
        del listt_days[index]
        del listt_price[index]
        del listt_num[index]

        dictionary["First_Name"] = None
        dictionary["First_Name"] = listt_fname

        dictionary["Last_Name"] = None
        dictionary["Last_Name"] = listt_lname

        dictionary["Phone_num"] = None
        dictionary["Phone_num"] = listt_phone

        dictionary["Room_Type"] = None
        dictionary["Room_Type"] = listt_type

        dictionary["Days"] = None
        dictionary["Days"] = listt_days

        dictionary["Price"] = None
        dictionary["Price"] = listt_price

        dictionary["Room"] = None
        dictionary["Room"] = listt_num

        with open("Management.txt", "w", encoding="utf-8") as file1:
            file1.write(str(dictionary))

        print("Details has been removed successfully")

        exit_menu()


def view():

    with open("Management.txt") as File:
        dictionary = json.loads(File.read().replace("'", '"'))

    dict_num = dictionary.get("Room")
    dict_len = len(dict_num)
    if dict_len == 0:
        print("\nThere is no data in our database\n")
        menu()

    else:
        listt = dictionary["Room"]
        a = len(listt)

        index = 0
        while index != a:
            listt_fname = dictionary.get("First_Name")
            listt_lname = dictionary.get("Last_Name")
            listt_phone = dictionary.get("Phone_num")
            listt_type = dictionary.get("Room_Type")
            listt_days = dictionary.get("Days")
            listt_price = dictionary.get("Price")
            listt_num = dictionary.get("Room")

            print("")
            print("First Name:", listt_fname[index])
            print("Last Name:", listt_lname[index])
            print("Phone number:", listt_phone[index])
            print("Room Type:", listt_type[index])
            print("Days staying:", listt_days[index])
            print("Money paid:", listt_price[index])
            print("Room Number:", listt_num[index])
            print("")

            index = index + 1

        exit_menu()


def exit():
    print("")
    print("                             Thanks for visiting")
    print("                                 Goodbye")


def exit_menu():
    print("")
    print("Do you want to exit the program or return to main menu")
    print("1-Main Menu")
    print("2-Exit")
    print("")

    user_input = int(input("Enter your choice: "))
    if user_input == 2:
        exit()
    elif user_input == 1:
        menu()


try:
    menu()
except KeyboardInterrupt as exit:
    print("\nexiting...!")

# menu()
