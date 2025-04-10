# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
##open project
# -

# # data is abstract Part :)

data = {
    "accno": [1001, 1002, 1003, 1004, 1005],
    "name": ["vaibhav", "abhinav", "aman", "ashish", "pramod"],
    "balance": [10000, 12000, 7000, 9000, 10000],
    "password": ["admin", "adminadmin", "passwd", "1234567", "amigo"],
    "security_check": ["2211", "1112", "1009", "1307", "1103"],
}

data

# import getpass
print("-------------------".center(100))
print("| Bank Application |".center(100))
print("-" * 100)
while True:
    print("\n 1. Login \n 2. Signup \n 3. Exit")
    i1 = int(input("enter what you want login, signup, exit :".center(50)))
    # login part
    if i1 == 1:
        print("login".center(90))
        print("_" * 100)
        i2 = int(input("enter account number : ".center(50)))
        if i2 in (data["accno"]):
            check = (data["accno"]).index(i2)
            i3 = input("enter password : ".center(50))
            check2 = data["password"].index(i3)
            if check == check2:
                while True:
                    print(
                        "\n 1.check ditails \n 2. debit \n 3. credit \n 4. change password \n 5. main Manu "
                    )
                    i4 = int(input("enter what you want :".center(50)))
                    # check ditails part
                    if i4 == 1:
                        print("cheak ditails".center(90))
                        print("." * 100)
                        print(f"your account number -->  {data['accno'][check]}")
                        print(f"your name -->  {data['name'][check]}")
                        print(f"your balance -->  {data['balance'][check]}")
                        continue
                    # debit part
                    elif i4 == 2:
                        print("debit".center(90))
                        print("." * 100)
                        print(f"your balance -->  {data['balance'][check]}")
                        i5 = int(input("enter debit amount : "))
                        if 0 < i5 <= data["balance"][check]:
                            debit = data["balance"][check] - i5
                            data["balance"][check] = debit
                            print(
                                f"your remaining balance -->  {data['balance'][check]}"
                            )
                        else:
                            print("your debit amount is more than balance ")
                        continue
                    # credit part
                    elif i4 == 3:
                        print("credit".center(90))
                        print("." * 100)
                        print(f"your balance -->  {data['balance'][check]}")
                        i6 = int(input("enter credit amount : "))
                        if 0 < i6:
                            credit = data["balance"][check] + i6
                            data["balance"][check] = credit
                            print(f"your new balance -->  {data['balance'][check]}")
                        else:
                            print("your credit amount is low ")
                        continue
                    # password part
                    elif i4 == 4:
                        print("change password".center(90))
                        print("." * 100)
                        old = input("enter your old password : ")
                        print(
                            "your password must have at list one lower case, one uppercase, one digital, one special case and length of password is 8"
                        )
                        new = getpass.getpass(prompt="Enter your new password")
                        if old == data["password"][check]:
                            low, up, sp, di = 0, 0, 0, 0
                            if (len(new)) > 8:
                                for i in new:
                                    if i.islower():
                                        low += 1
                                    if i.isupper():
                                        up += 1
                                    if i.isdigit():
                                        di += 1
                                    if i in ["@", "$", "%", "^", "&", "*"]:
                                        sp += 1
                            if (
                                low >= 1
                                and up >= 1
                                and sp >= 1
                                and di >= 1
                                and low + up + sp + di == len(new)
                            ):
                                data["password"][check] = new
                                print(
                                    f"your new password -->  {data['password'][check]}"
                                )
                            else:
                                print("Invalid Password")
                        else:
                            print("old password wrong please enter valid password")
                        continue
                    elif i4 == 5:
                        print("main menu".center(90))
                        print("." * 100)
                    break
                else:
                    print("please enter valid number")
            else:
                print("please check your password number".center(50))
        else:
            print("please check your account number".center(50))
    # signup part
    elif i1 == 2:
        print("signup".center(90))
        print("_" * 100)
        acc = 1001 + len(data["accno"])
        data["accno"].append(acc)
        ind = (data["accno"]).index(acc)
        name = input("enter your name : ")
        data["name"].append(name)
        balance = int(input("enter your initial balance : "))
        data["balance"].append(balance)
        password = input("enter your password : ")
        data["password"].append(password)
        security_check = (int(input("enter your security pin (DDMM) : "))).split()
        print("." * 100)
        print(f"your account number -->  {data['accno'][ind]}".center(50))
        print(f"your name -->  {data['name'][ind]}".center(50))
        print(f"your balance -->  {data['balance'][ind]}".center(50))
        print(f"your password --> {data['password'][ind]}".center(50))
        continue
    # exit part
    elif i1 == 3:
        print("exit".center(90))
        print("thank you for visiting".center(90))
        print("." * 100)
        break
    else:
        print(f"wrong enter : {i1}".center(50))


# # All part in function:)

def cheak_ditails(check):
    print("cheak ditails".center(90))
    print("." * 100)
    print(f"your account number -->  {data['accno'][check]}")
    print(f"your name -->  {data['name'][check]}")
    print(f"your balance -->  {data['balance'][check]}")


def credit(check):
    print("credit".center(90))
    print("." * 100)
    print(f"your balance -->  {data['balance'][check]}")
    i6 = int(input("enter credit amount : "))
    if 0 < i6:
        credit = data["balance"][check] + i6
        data["balance"][check] = credit
        print(f"your new balance -->  {data['balance'][check]}")
    else:
        print("your credit amount is low ")


def debit(check):
    print("debit".center(90))
    print("." * 100)
    print(f"your balance -->  {data['balance'][check]}")
    i5 = int(input("enter debit amount : "))
    if 0 < i5 <= data["balance"][check]:
        debit = data["balance"][check] - i5
        data["balance"][check] = debit
        print(f"your remaining balance -->  {data['balance'][check]}")
    else:
        print("your debit amount is more than balance ")


def change_password(check):
    print("change password".center(90))
    print("." * 100)
    old = input("enter your old password : ")
    print(
        "your password must have at list one lower case, one uppercase, one digital, one special case and length of password is 8"
    )
    new = getpass.getpass(prompt="Enter your new password")
    if old == data["password"][check]:
        low, up, sp, di = 0, 0, 0, 0
        if (len(new)) > 8:
            for i in new:
                if i.islower():
                    low += 1
                if i.isupper():
                    up += 1
                if i.isdigit():
                    di += 1
                if i in ["@", "$", "%", "^", "&", "*"]:
                    sp += 1
        if (
            low >= 1
            and up >= 1
            and sp >= 1
            and di >= 1
            and low + up + sp + di == len(new)
        ):
            data["password"][check] = new
            print(f"your new password -->  {data['password'][check]}")
        else:
            print("Invalid Password")
    else:
        print("old password wrong please enter valid password")


def login():
    print("login".center(90))
    print("_" * 100)
    i2 = int(input("enter account number : ".center(50)))
    if i2 in (data["accno"]):
        check = (data["accno"]).index(i2)
        i3 = input("enter password : ".center(50))
        check2 = data["password"].index(i3)
        if check == check2:
            while True:
                print(
                    "\n 1.check ditails \n 2. debit \n 3. credit \n 4. change password \n 5. main Manu "
                )
                i4 = int(input("enter what you want :".center(50)))
                # check ditails part
                if i4 == 1:
                    cheak_ditails(check)
                    continue
                # debit part
                elif i4 == 2:
                    debit(check)
                    continue
                # credit part
                elif i4 == 3:
                    credit(check)
                    continue
                # password part
                elif i4 == 4:
                    change_password(check)
                    continue
                elif i4 == 5:
                    print("main menu".center(90))
                    print("." * 100)
                break
            else:
                print("please enter valid number")
        else:
            print("please check your password number".center(50))
    else:
        print("please check your account number".center(50))


def security_check(ss):
    data = {(1, 3, 5, 7, 8, 10, 12): 31, (2,): 29, (4, 6, 9): 30}
    month = ss[2:]
    date = ss[:2]
    for key, value in data.items():
        print(key, value)
        if int(month) in key:
            if 1 <= int(date) <= value:
                return True
            return False
    return False


def signup():
    print("signup".center(90))
    print("_" * 100)
    acc = 1001 + len(data["accno"])
    data["accno"].append(acc)
    ind = (data["accno"]).index(acc)
    name = input("enter your name : ")
    data["name"].append(name)
    balance = int(input("enter your initial balance : "))
    data["balance"].append(balance)
    password = input("enter your password : ")
    data["password"].append(password)
    ss = input("enter a secuirty quetion in form dd//mm")
    security_check(ss)
    data["security_check"].append(ss)
    print("." * 100)
    print(f"your account number -->  {data['accno'][ind]}".center(50))
    print(f"your name -->  {data['name'][ind]}".center(50))


def main():
    import getpass

    print("-------------------".center(100))
    print("| Bank Application |".center(100))
    print("-" * 100)
    while True:
        print("\n 1. Login \n 2. Signup \n 3. Exit")
        i1 = int(input("enter what you want login, signup, exit :".center(50)))
        # login part
        if i1 == 1:
            login()
        # signup part
        elif i1 == 2:
            signup()
        # exit part
        elif i1 == 3:
            print("exit".center(90))
            print("thank you for visiting".center(90))
            print("." * 100)
            break
        else:
            print(f"wrong enter : {i1}".center(50))


