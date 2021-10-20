import sqlite3


# making connection with database
def connect_database():
    global conn
    global cur
    conn = sqlite3.connect("bankmanaging.db")

    cur = conn.cursor()

    cur.execute(
        "create table if not exists bank (acc_no int, name text, age int, address text, balance int, account_type text, mobile_number int)")
    cur.execute("create table if not exists staff (name text, pass text,salary int, position text)")
    cur.execute("create table if not exists admin (name text, pass text)")
    cur.execute("insert into admin values('arpit','123')")
    conn.commit()
    cur.execute("select acc_no from bank")
    acc = cur.fetchall()
    global acc_no
    if len(acc) == 0:
        acc_no = 1
    else:
        acc_no = int(acc[-1][0]) + 1


# check admin dtails in database
def check_admin(name, password):
    cur.execute("select * from admin")
    data = cur.fetchall()

    if data[0][0] == name and data[0][1] == password:
        return True
    return


# create employee in database
def create_employee(name, password, salary, positon):
    print(password)
    cur.execute("insert into staff values(?,?,?,?)", (name, password, salary, positon))
    conn.commit()


# check employee details in dabase for employee login
def check_employee(name, password):
    print(password)
    print(name)
    cur.execute("select name,pass from staff")
    data = cur.fetchall()
    print(data)
    if len(data) == 0:
        return False
    for i in range(len(data)):
        if data[i][0] == name and data[i][1] == password:
            return True

    return False


# create customer details in database
def create_customer(name, age, address, balance, acc_type, mobile_number):
    global acc_no
    cur.execute("insert into bank values(?,?,?,?,?,?,?)",
                (acc_no, name, age, address, balance, acc_type, mobile_number))
    conn.commit()
    acc_no = acc_no + 1
    return acc_no - 1


# check account in database
def check_acc_no(acc_no):
    cur.execute("select acc_no from bank")
    list_acc_no = cur.fetchall()

    for i in range(len(list_acc_no)):
        if list_acc_no[i][0] == int(acc_no):
            return True
    return False


# get all details of a particular customer from database
def get_details(acc_no):
    cur.execute("select * from bank where acc_no=?", (acc_no))
    global detail
    detail = cur.fetchall()
    print(detail)
    if len(detail) == 0:
        return False
    else:
        return (detail[0][0], detail[0][1], detail[0][2], detail[0][3], detail[0][4], detail[0][5], detail[0][6])


# add new balance of customer in bank database
def update_balance(new_money, acc_no):
    cur.execute("select balance from bank where acc_no=?", (acc_no,))
    bal = cur.fetchall()
    bal = bal[0][0]
    new_bal = bal + int(new_money)

    cur.execute("update bank set balance=? where acc_no=?", (new_bal, acc_no))
    conn.commit()


# deduct balance from customer bank database
def deduct_balance(new_money, acc_no):
    cur.execute("select balance from bank where acc_no=?", (acc_no,))
    bal = cur.fetchall()
    bal = bal[0][0]
    if bal < int(new_money):
        return False
    else:
        new_bal = bal - int(new_money)

        cur.execute("update bank set balance=? where acc_no=?", (new_bal, acc_no))
        conn.commit()
        return True


# gave balance of a particular account number from database
def check_balance(acc_no):
    cur.execute("select balance from bank where acc_no=?", (acc_no))
    bal = cur.fetchall()
    return bal[0][0]


# update_name_in_bank_table
def update_name_in_bank_table(new_name, acc_no):
    print(new_name)
    conn.execute("update bank set name='{}' where acc_no={}".format(new_name, acc_no))
    conn.commit()


# update_age_in_bank_table
def update_age_in_bank_table(new_name, acc_no):
    print(new_name)
    conn.execute("update bank set age={} where acc_no={}".format(new_name, acc_no))
    conn.commit()


# update_address_in_bank_table
def update_address_in_bank_table(new_name, acc_no):
    print(new_name)
    conn.execute("update bank set address='{}' where acc_no={}".format(new_name, acc_no))
    conn.commit()


# list of all customers in bank
def list_all_customers():
    cur.execute("select * from bank")
    deatil = cur.fetchall()

    return deatil


# delete account from database
def delete_acc(acc_no):
    cur.execute("delete from bank where acc_no=?", (acc_no))
    conn.commit()


# show employees detail from staff table
def show_employees():
    cur.execute("select name, salary, position,pass from staff")
    detail = cur.fetchall()
    return detail


# return all money in bank
def all_money():
    cur.execute("select balance from bank")
    bal = cur.fetchall()
    print(bal)
    if len(bal) == 0:
        return False
    else:
        total = 0
        for i in bal:
            total = total + i[0]
        return total


# return a list of all employees name
def show_employees_for_update():
    cur.execute("select * from staff")
    detail = cur.fetchall()
    return detail


# update employee name from data base
def update_employee_name(new_name, old_name):
    print(new_name, old_name)
    cur.execute("update staff set name='{}' where name='{}'".format(new_name, old_name))
    conn.commit()


def update_employee_password(new_pass, old_name):
    print(new_pass, old_name)
    cur.execute("update staff set pass='{}' where name='{}'".format(new_pass, old_name))
    conn.commit()


def update_employee_salary(new_salary, old_name):
    print(new_salary, old_name)
    cur.execute("update staff set salary={} where name='{}'".format(new_salary, old_name))
    conn.commit()


def update_employee_position(new_pos, old_name):
    print(new_pos, old_name)
    cur.execute("update staff set position='{}' where name='{}'".format(new_pos, old_name))
    conn.commit()


# get name and balance from bank of a particular account number
def get_detail(acc_no):
    cur.execute("select name, balance from bank where acc_no=?", (acc_no))
    details = cur.fetchall()
    return details


def check_name_in_staff(name):
    cur = conn.cursor()
    cur.execute("select name from staff")
    details = cur.fetchall()

    for i in details:
        if i[0] == name:
            return True
    return False
