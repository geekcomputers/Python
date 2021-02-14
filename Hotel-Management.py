

def menu():
    print("")
    print("")
    print("                         Welcome to Hotel Database Management Software")
    print("")
    print("")

    print("1-Add new customer details")
    print("2-Modify already existing customer details")
    print("3-Search customer details")
    print("4-View all customer details")
    print("5-Delete customer details")
    print("6-Exit the program")
    print("")

    user_input=int(input("Enter your choice(1-6): "))

    if user_input==1:
        add()

    elif user_input==2:
        modify()

    elif user_input==3:
        search()

    elif user_input==4:
        view()

    elif user_input==5:
        remove()

    elif user_input==6:
        exit()

def add():

    print("")
    Name1=input("Enter your first name: ")
    print("")

    Name2=input("Enter your last name: ")
    print("")

    Phone_Num=input("Enter your phone number(without +91): ")
    print("")

    print("These are the rooms that are currently available")
    print("1-Normal (500/Day)")
    print("2-Deluxe (1000/Day)")
    print("3-Super Deluxe (1500/Day)")
    print("4-Premium Deluxe (2000/Day)")
    print("")
    Room_Type=int(input("Which type you want(1-4): "))
    print("")

    if Room_Type==1:
        x=500
        Room_Type="Normal"
    elif Room_Type==2:
        x=1000
        Room_Type='Deluxe'
    elif Room_Type==3:
        x=1500
        Room_Type='Super Deluxe'
    elif Room_Type==4:
        x=2000
        Room_Type='Premium'

    Days=int(input("How many days you will stay: "))
    Money=x*Days
    Money=str(Money)
    print("")

    print("You have to pay ",(Money))
    print("")


    Payment=input("Mode of payment(Card/Cash/Online): ")
    print("")


    File=open('Management.txt','r')
    string=File.read()
    string = string.replace("\'", "\"")
    dictionary=json.loads(string)
    File.close()


    if len(dictionary.get('Room'))==0:
        Room_num='501'
    else:
        listt=dictionary.get('Room')
        tempp=len(listt)-1
        temppp=int(listt[tempp])
        Room_num=(1+temppp)
        Room_num=str(Room_num)

    print('You have been assigned Room Number',Room_num)

    dictionary['First_Name'].append(Name1)
    dictionary['Last_Name'].append(Name2)
    dictionary['Phone_num'].append(Phone_Num)
    dictionary['Room_Type'].append(Room_Type)
    dictionary['Days'].append(Days)
    dictionary['Price'].append(Money)
    dictionary['Room'].append(Room_num)

    File=open("Management.txt",'w',encoding="utf-8")
    File.write(str(dictionary))
    File.close()

    print("")
    print("Your data has been successfully added to our database.")

    exit_menu()



import os
import json
filecheck = os.path.isfile('Management.txt')
if filecheck == False :
    File = open("Management.txt", 'a', encoding="utf-8")
    temp1 = {'First_Name': [], 'Last_Name': [], 'Phone_num': [], 'Room_Type': [], 'Days': [], 'Price': [], 'Room':[]}
    File.write(str(temp1))
    File.close()



def modify():

    File=open('Management.txt','r')
    string=File.read()
    string = string.replace("\'", "\"")
    dictionary=json.loads(string)
    File.close()

    dict_num=dictionary.get("Room")
    dict_len=len(dict_num)
    if dict_len==0:
        print("")
        print("There is no data in our database")
        print("")
        menu()
    else:
        print("")
        Room=(input("Enter your Room Number: "))

        listt=dictionary['Room']
        index=int(listt.index(Room))

        print("")
        print("1-Change your first name")
        print("2-Change your last name")
        print("3-Change your phone number")

        print("")
        choice=(input("Enter your choice: "))
        print("")

        File=open("Management.txt",'w',encoding="utf-8")

        if choice == str(1):
            user_input=input('Enter New First Name: ')
            listt1=dictionary['First_Name']
            listt1[index]=user_input
            dictionary['First_Name']=None
            dictionary['First_Name']=listt1
            File.write(str(dictionary))
            File.close()

        elif choice == str(2):
            user_input = input('Enter New Last Name: ')
            listt1 = dictionary['Last_Name']
            listt1[index] = user_input
            dictionary['Last_Name'] = None
            dictionary['Last_Name'] = listt1
            File.write(str(dictionary))
            File.close()

        elif choice == str(3):
            user_input = input('Enter New Phone Number: ')
            listt1 = dictionary['Phone_num']
            listt1[index] = user_input
            dictionary['Phone_num'] = None
            dictionary['Phone_num'] = listt1
            File.write(str(dictionary))
            File.close()

        print("")
        print("Your data has been successfully updated")

        exit_menu()

def search():

    File=open('Management.txt','r')
    string=File.read()
    string = string.replace("\'", "\"")
    dictionary=json.loads(string)
    File.close()

    dict_num=dictionary.get("Room")
    dict_len=len(dict_num)
    if dict_len==0:
        print("")
        print("There is no data in our database")
        print("")
        menu()
    else:
        print("")
        Room = (input("Enter your Room Number: "))
        print("")

        listt = dictionary['Room']
        index = int(listt.index(Room))

        listt_fname=dictionary.get('First_Name')
        listt_lname=dictionary.get('Last_Name')
        listt_phone=dictionary.get('Phone_num')
        listt_type=dictionary.get('Room_Type')
        listt_days=dictionary.get('Days')
        listt_price=dictionary.get('Price')
        listt_num=dictionary.get('Room')

        print("")
        print("First Name:",listt_fname[index])
        print("Last Name:",listt_lname[index])
        print("Phone number:",listt_phone[index])
        print("Room Type:",listt_type[index])
        print('Days staying:',listt_days[index])
        print('Money paid:',listt_price[index])
        print('Room Number:',listt_num[index])

        exit_menu()

def remove():
    File=open('Management.txt','r')
    string=File.read()
    string = string.replace("\'", "\"")
    dictionary=json.loads(string)
    File.close()

    dict_num=dictionary.get("Room")
    dict_len=len(dict_num)
    if dict_len==0:
        print("")
        print("There is no data in our database")
        print("")
        menu()
    else:
        print("")
        Room = (input("Enter your Room Number: "))
        print("")

        listt = dictionary['Room']
        index = int(listt.index(Room))

        listt_fname = dictionary.get('First_Name')
        listt_lname = dictionary.get('Last_Name')
        listt_phone = dictionary.get('Phone_num')
        listt_type = dictionary.get('Room_Type')
        listt_days = dictionary.get('Days')
        listt_price = dictionary.get('Price')
        listt_num = dictionary.get('Room')

        del listt_fname[index]
        del listt_lname[index]
        del listt_phone[index]
        del listt_type[index]
        del listt_days[index]
        del listt_price[index]
        del listt_num[index]

        dictionary['First_Name'] = None
        dictionary['First_Name'] = listt_fname

        dictionary['Last_Name']= None
        dictionary['Last_Name']= listt_lname

        dictionary['Phone_num']= None
        dictionary['Phone_num']=listt_phone

        dictionary['Room_Type']=None
        dictionary['Room_Type']=listt_type

        dictionary['Days']=None
        dictionary['Days']=listt_days

        dictionary['Price']=None
        dictionary['Price']=listt_price

        dictionary['Room']=None
        dictionary['Room']=listt_num

        file1=open('Management.txt','w',encoding="utf-8")
        file1.write(str(dictionary))
        file1.close()

        print("Details has been removed successfully")

        exit_menu()

def view():

    File=open('Management.txt','r')
    string=File.read()
    string = string.replace("\'", "\"")
    dictionary=json.loads(string)
    File.close()

    dict_num=dictionary.get("Room")
    dict_len=len(dict_num)
    if dict_len==0:
        print("")
        print("There is no data in our database")
        print("")
        menu()

    else:
        listt = dictionary['Room']
        a = len(listt)

        index=0
        while index!=a:
            listt_fname = dictionary.get('First_Name')
            listt_lname = dictionary.get('Last_Name')
            listt_phone = dictionary.get('Phone_num')
            listt_type = dictionary.get('Room_Type')
            listt_days = dictionary.get('Days')
            listt_price = dictionary.get('Price')
            listt_num = dictionary.get('Room')

            print("")
            print("First Name:", listt_fname[index])
            print("Last Name:", listt_lname[index])
            print("Phone number:", listt_phone[index])
            print("Room Type:", listt_type[index])
            print('Days staying:', listt_days[index])
            print('Money paid:', listt_price[index])
            print('Room Number:', listt_num[index])
            print("")

            index=index+1

        exit_menu()

def exit():
    print("")
    print('                             Thanks for visiting')
    print("                                 Goodbye")

def exit_menu():
    print("")
    print("Do you want to exit the program or return to main menu")
    print("1-Main Menu")
    print("2-Exit")
    print("")

    user_input=int(input("Enter your choice: "))
    if user_input==2:
        exit()
    elif user_input==1:
        menu()

menu()