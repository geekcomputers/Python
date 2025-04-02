import sqlite3

# Making connection with database
def connect_database():
    global conn
    global cur
    conn = sqlite3.connect("bankmanaging.db")
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bank (
            acc_no INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            address TEXT,
            balance INTEGER,
            account_type TEXT,
            mobile_number TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS staff (
            name TEXT,
            pass TEXT,
            salary INTEGER,
            position TEXT
        )
        """
    )
    cur.execute("CREATE TABLE IF NOT EXISTS admin (name TEXT, pass TEXT)")

    # Only insert admin if not exists
    cur.execute("SELECT COUNT(*) FROM admin")
    if cur.fetchone()[0] == 0:
        cur.execute("INSERT INTO admin VALUES (?, ?)", ('arpit', '123'))
    
    conn.commit()

    # Fetch last account number to avoid duplicate or incorrect numbering
    cur.execute("SELECT acc_no FROM bank ORDER BY acc_no DESC LIMIT 1")
    acc = cur.fetchone()
    global acc_no
    acc_no = 1 if acc is None else acc[0] + 1

# Check admin details in database
def check_admin(name, password):
    cur.execute("SELECT * FROM admin WHERE name = ? AND pass = ?", (name, password))
    return cur.fetchone() is not None

# Create employee in database
def create_employee(name, password, salary, position):
    cur.execute("INSERT INTO staff VALUES (?, ?, ?, ?)", (name, password, salary, position))
    conn.commit()

# Check employee login details
def check_employee(name, password):
    cur.execute("SELECT 1 FROM staff WHERE name = ? AND pass = ?", (name, password))
    return cur.fetchone() is not None

# Create customer in database
def create_customer(name, age, address, balance, acc_type, mobile_number):
    global acc_no
    cur.execute(
        "INSERT INTO bank VALUES (?, ?, ?, ?, ?, ?, ?)",
        (acc_no, name, age, address, balance, acc_type, mobile_number)
    )
    conn.commit()
    acc_no += 1
    return acc_no - 1

# Check if account number exists
def check_acc_no(acc_no):
    cur.execute("SELECT 1 FROM bank WHERE acc_no = ?", (acc_no,))
    return cur.fetchone() is not None

# Get customer details
def get_details(acc_no):
    cur.execute("SELECT * FROM bank WHERE acc_no = ?", (acc_no,))
    detail = cur.fetchone()
    return detail if detail else False

# Update customer balance
def update_balance(new_money, acc_no):
    cur.execute("UPDATE bank SET balance = balance + ? WHERE acc_no = ?", (new_money, acc_no))
    conn.commit()

# Deduct balance
def deduct_balance(new_money, acc_no):
    cur.execute("SELECT balance FROM bank WHERE acc_no = ?", (acc_no,))
    bal = cur.fetchone()
    if bal and bal[0] >= new_money:
        cur.execute("UPDATE bank SET balance = balance - ? WHERE acc_no = ?", (new_money, acc_no))
        conn.commit()
        return True
    return False

# Get account balance
def check_balance(acc_no):
    cur.execute("SELECT balance FROM bank WHERE acc_no = ?", (acc_no,))
    bal = cur.fetchone()
    return bal[0] if bal else 0

# Update customer details
def update_name_in_bank_table(new_name, acc_no):
    cur.execute("UPDATE bank SET name = ? WHERE acc_no = ?", (new_name, acc_no))
    conn.commit()

def update_age_in_bank_table(new_age, acc_no):
    cur.execute("UPDATE bank SET age = ? WHERE acc_no = ?", (new_age, acc_no))
    conn.commit()

def update_address_in_bank_table(new_address, acc_no):
    cur.execute("UPDATE bank SET address = ? WHERE acc_no = ?", (new_address, acc_no))
    conn.commit()

# List all customers
def list_all_customers():
    cur.execute("SELECT * FROM bank")
    return cur.fetchall()

# Delete account
def delete_acc(acc_no):
    cur.execute("DELETE FROM bank WHERE acc_no = ?", (acc_no,))
    conn.commit()

# Show employees
def show_employees():
    cur.execute("SELECT name, salary, position FROM staff")
    return cur.fetchall()

# Get total money in bank
def all_money():
    cur.execute("SELECT SUM(balance) FROM bank")
    total = cur.fetchone()[0]
    return total if total else 0

# Get employee details
def show_employees_for_update():
    cur.execute("SELECT * FROM staff")
    return cur.fetchall()

# Update employee details
def update_employee_name(new_name, old_name):
    cur.execute("UPDATE staff SET name = ? WHERE name = ?", (new_name, old_name))
    conn.commit()

def update_employee_password(new_pass, old_name):
    cur.execute("UPDATE staff SET pass = ? WHERE name = ?", (new_pass, old_name))
    conn.commit()

def update_employee_salary(new_salary, old_name):
    cur.execute("UPDATE staff SET salary = ? WHERE name = ?", (new_salary, old_name))
    conn.commit()

def update_employee_position(new_pos, old_name):
    cur.execute("UPDATE staff SET position = ? WHERE name = ?", (new_pos, old_name))
    conn.commit()

# Get customer name and balance
def get_detail(acc_no):
    cur.execute("SELECT name, balance FROM bank WHERE acc_no = ?", (acc_no,))
    return cur.fetchone()

# Check if employee exists
def check_name_in_staff(name):
    cur.execute("SELECT 1 FROM staff WHERE name = ?", (name,))
    return cur.fetchone() is not Non