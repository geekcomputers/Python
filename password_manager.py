import sqlite3
from getpass import getpass
import os

# set the environment variable ADMIN_PASS to your desired string, which will be your password.
ADMIN_PASSWORD = os.environ['ADMIN_PASS']
connect = getpass("What is your admin  password?\n")

while connect != ADMIN_PASSWORD:
    connect = getpass("What is your admin password?\n")
    if connect == "q":
        break

conn = sqlite3.connect('password_manager.db')
cursor_ = conn.cursor()


def get_password(service_):
    command = 'SELECT * from STORE WHERE SERVICE = "' + service_ + '"'
    cursor = conn.execute(command)
    for row in cursor:
        username_ = row[1]
        password_ = row[2]
    return [username_, password_]


def add_password(service_, username_, password_):
    command = 'INSERT INTO STORE (SERVICE,USERNAME,PASSWORD) VALUES("'+service_+'","'+username_+'","'+password_+'");'
    conn.execute(command)
    conn.commit()


def update_password(service_, password_):
    command = 'UPDATE STORE set PASSWORD = "' + password_ + '" where SERVICE = "' + service_ + '"'
    conn.execute(command)
    conn.commit()
    print(service_ + " password updated successfully.")


def delete_service(service_):
    command = 'DELETE from STORE where SERVICE = "' + service_ + '"'
    conn.execute(command)
    conn.commit()
    print(service_ + " deleted from the database successfully.")


def get_all():
    cursor_.execute("SELECT * from STORE")
    data = cursor_.fetchall()
    if len(data) == 0:
        print('No Data')
    else:
        for row in data:
            print("service = ", row[0])
            print("username = ", row[1])
            print("password = ", row[2])
            print()


def is_service_present(service_):
    cursor_.execute("SELECT SERVICE from STORE where SERVICE = ?", (service_,))
    data = cursor_.fetchall()
    if len(data) == 0:
        print('There is no service named %s' % service_)
        return False
    else:
        return True


if connect == ADMIN_PASSWORD:
    try:
        conn.execute('''CREATE TABLE STORE
            (SERVICE TEXT PRIMARY KEY NOT NULL,
            USERNAME TEXT NOT NULL,
            PASSWORD TEXT NOT NULL);
            ''')
        print("Your safe has been created!\nWhat would you like to store in it today?")
    except:
        print("You have a safe, what would you like to do today?")

    while True:
        print("\n" + "*" * 15)
        print("Commands:")
        print("quit = quit program")
        print("get = get username and password")
        print("getall = show all the details in the database")
        print("store = store username and password")
        print("update = update password")
        print("delete = delete a service details")
        print("*" * 15)
        input_ = input(":")

        if input_ == "quit":
            print("\nGoodbye, have a great day.\n")
            conn.close()
            break

        elif input_ == "store":
            service = input("What is the name of the service?\n")
            cursor_.execute("SELECT SERVICE from STORE where SERVICE = ?", (service,))
            data = cursor_.fetchall()
            if len(data) == 0:
                username = input("Enter username : ")
                password = getpass("Enter password : ")
                if username == '' or password == '':
                    print("Your username or password is empty.")
                else:
                    add_password(service, username, password)
                    print("\n" + service.capitalize() + " password stored\n")
            else:
                print("Service named {} already exists.".format(service))

        elif input_ == "get":
            service = input("What is the name of the service?\n")
            flag = is_service_present(service)
            if flag:
                username, password = get_password(service)
                print(service.capitalize() + " Details")
                print("Username : ", username)
                print("Password : ", password)

        elif input_ == "update":
            service = input("What is the name of the service?\n")
            if service == '':
                print('Service is not entered.')
            else:
                flag = is_service_present(service)
                if flag:
                    password = getpass("Enter new password : ")
                    update_password(service, password)

        elif input_ == "delete":
            service = input("What is the name of the service?\n")
            if service == '':
                print('Service is not entered.')
            else:
                flag = is_service_present(service)
                if flag:
                    delete_service(service)

        elif input_ == "getall":
            get_all()

        else:
            print("Invalid command.")
