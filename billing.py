updated_billing
items = {"apple": 5, "soap": 4, "soda": 6, "pie": 7, "cake": 20}
total_price = 0
try:
    print("""
Press 1 for apple
Press 2 for soap
Press 3 for soda
Press 4 for pie
Press 5 for cake
Press 6 for bill""")
    while True:
        choice = int(input("enter your choice here..\n"))
        if choice == 1:
            print("Apple added to the cart")
            total_price += items["apple"]

        elif choice == 2:
            print("soap added to the cart")
            total_price += items["soap"]
        elif choice == 3:
            print("soda added to the cart")
            total_price += items["soda"]
        elif choice == 4:
            print("pie added to the cart")
            total_price += items["pie"]
        elif choice == 5:
            print("cake added to the cart")
            total_price += items["cake"]
        elif choice == 6:
            print(f"""

Total amount :{total_price}
""")
            break
        else:
            print("Please enter the digits within the range 1-6..")
except:
    print("enter only digits")

"""
Code Explanation:
A dictionary named items is created to store product names and their corresponding prices.
Example: "apple": 5 means apple costs 5 units.

one variable is initialized:

total_price to keep track of the overall bill.


A menu is printed that shows the user what number to press for each item or to generate the final bill.

A while True loop is started, meaning it will keep running until the user explicitly chooses to stop (by selecting "6" for the bill).

Inside the loop:

The user is asked to enter a number (1–6).

Depending on their input:

If they enter 1–5, the corresponding item is "added to the cart" and its price is added to the total_price.

If they enter 6, the total price is printed and the loop breaks (ends).

If they enter something outside 1–6, a warning message is shown.

The try-except block is used to catch errors if the user enters something that's not a number (like a letter or symbol).
In that case, it simply shows: "enter only digits".
"""
