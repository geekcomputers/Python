"""
Bank Management System - Database Module

This module provides database operations for a banking system,
including customer and employee management, account operations,
and administrative functions.
"""

import os
import sqlite3

# Database connection and cursor
conn: sqlite3.Connection
cur: sqlite3.Cursor
acc_no: int

def connect_database() -> None:
    """
    Connect to the SQLite database and initialize tables if they don't exist.
    
    Creates the following tables:
    - bank: Stores customer account information
    - staff: Stores employee details
    - admin: Stores administrator credentials
    
    Initializes the first account number and creates a default admin user if none exists.
    """
    global conn, cur, acc_no
    
    # Connect to database and create tables
    db_path = os.path.join(os.path.dirname(__file__), "bankmanaging.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Create bank table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bank (
            acc_no INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            address TEXT,
            balance INTEGER,
            account_type TEXT,
            mobile_number TEXT
        )
    """)
    
    # Create staff table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS staff (
            name TEXT,
            pass TEXT,
            salary INTEGER,
            position TEXT
        )
    """)
    
    # Create admin table
    cur.execute("CREATE TABLE IF NOT EXISTS admin (name TEXT, pass TEXT)")
    
    # Initialize default admin if none exists
    cur.execute("SELECT COUNT(*) FROM admin")
    if cur.fetchone()[0] == 0:
        cur.execute("INSERT INTO admin VALUES (?, ?)", ('admin', 'admin123'))
    
    conn.commit()
    
    # Initialize next account number
    cur.execute("SELECT acc_no FROM bank ORDER BY acc_no DESC LIMIT 1")
    acc = cur.fetchone()
    acc_no = 1 if acc is None else acc[0] + 1

def check_admin(name: str, password: str) -> bool:
    """Verify administrator credentials."""
    cur.execute("SELECT * FROM admin WHERE name = ? AND pass = ?", (name, password))
    return cur.fetchone() is not None

def create_employee(name: str, password: str, salary: int, position: str) -> None:
    """Create a new staff member."""
    cur.execute("INSERT INTO staff VALUES (?, ?, ?, ?)", (name, password, salary, position))
    conn.commit()

def check_employee(name: str, password: str) -> bool:
    """Verify employee credentials."""
    cur.execute("SELECT 1 FROM staff WHERE name = ? AND pass = ?", (name, password))
    return cur.fetchone() is not None

def create_customer(
    name: str,
    age: int,
    address: str,
    balance: int,
    acc_type: str,
    mobile_number: str
) -> int:
    """Create a new customer account and return the account number."""
    global acc_no
    cur.execute(
        "INSERT INTO bank VALUES (?, ?, ?, ?, ?, ?, ?)",
        (acc_no, name, age, address, balance, acc_type, mobile_number)
    )
    conn.commit()
    account_created = acc_no
    acc_no += 1
    return account_created

def check_acc_no(acc_no: int) -> bool:
    """Check if an account number exists."""
    cur.execute("SELECT 1 FROM bank WHERE acc_no = ?", (acc_no,))
    return cur.fetchone() is not None

def get_details(acc_no: int) -> tuple[int, str, int, str, int, str, str] | None:
    """Retrieve full details of a customer account."""
    cur.execute("SELECT * FROM bank WHERE acc_no = ?", (acc_no,))
    return cur.fetchone()

def update_balance(new_money: int, acc_no: int) -> None:
    """Update account balance by adding/subtracting an amount."""
    cur.execute("UPDATE bank SET balance = balance + ? WHERE acc_no = ?", (new_money, acc_no))
    conn.commit()

def deduct_balance(new_money: int, acc_no: int) -> bool:
    """Deduct an amount from an account if sufficient balance exists."""
    cur.execute("SELECT balance FROM bank WHERE acc_no = ?", (acc_no,))
    bal = cur.fetchone()
    if bal and bal[0] >= new_money:
        cur.execute("UPDATE bank SET balance = balance - ? WHERE acc_no = ?", (new_money, acc_no))
        conn.commit()
        return True
    return False

def check_balance(acc_no: int) -> int:
    """Get the current balance of an account."""
    cur.execute("SELECT balance FROM bank WHERE acc_no = ?", (acc_no,))
    bal = cur.fetchone()
    return bal[0] if bal else 0

def update_name_in_bank_table(new_name: str, acc_no: int) -> None:
    """Update the name associated with an account."""
    cur.execute("UPDATE bank SET name = ? WHERE acc_no = ?", (new_name, acc_no))
    conn.commit()

def update_age_in_bank_table(new_age: int, acc_no: int) -> None:
    """Update the age associated with an account."""
    cur.execute("UPDATE bank SET age = ? WHERE acc_no = ?", (new_age, acc_no))
    conn.commit()

def update_address_in_bank_table(new_address: str, acc_no: int) -> None:
    """Update the address associated with an account."""
    cur.execute("UPDATE bank SET address = ? WHERE acc_no = ?", (new_address, acc_no))
    conn.commit()

def update_mobile_number_in_bank_table(new_mobile_number: str, acc_no: int) -> None:
    """Update the mobile number associated with an account."""
    cur.execute("UPDATE bank SET mobile_number = ? WHERE acc_no = ?", (new_mobile_number, acc_no))
    conn.commit()

def update_acc_type_in_bank_table(new_acc_type: str, acc_no: int) -> None:
    """Update the account type associated with an account."""
    cur.execute("UPDATE bank SET account_type = ? WHERE acc_no = ?", (new_acc_type, acc_no))
    conn.commit()

def list_all_customers() -> list[tuple[int, str, int, str, int, str, str]]:
    """Retrieve details of all customer accounts."""
    cur.execute("SELECT * FROM bank")
    return cur.fetchall()

def delete_acc(acc_no: int) -> None:
    """Delete a customer account."""
    cur.execute("DELETE FROM bank WHERE acc_no = ?", (acc_no,))
    conn.commit()

def show_employees() -> list[tuple[str, int, str]]:
    """Retrieve basic details of all employees."""
    cur.execute("SELECT name, salary, position FROM staff")
    return cur.fetchall()

def all_money() -> int:
    """Calculate the total balance across all customer accounts."""
    cur.execute("SELECT SUM(balance) FROM bank")
    total = cur.fetchone()[0]
    return total if total is not None else 0

def show_employees_for_update() -> list[tuple[str, str, int, str]]:
    """Retrieve complete details of all employees for update operations."""
    cur.execute("SELECT * FROM staff")
    return cur.fetchall()

def update_employee_name(new_name: str, old_name: str) -> None:
    """Update an employee's name."""
    cur.execute("UPDATE staff SET name = ? WHERE name = ?", (new_name, old_name))
    conn.commit()

def update_employee_password(new_pass: str, old_name: str) -> None:
    """Update an employee's password."""
    cur.execute("UPDATE staff SET pass = ? WHERE name = ?", (new_pass, old_name))
    conn.commit()

def update_employee_salary(new_salary: int, old_name: str) -> None:
    """Update an employee's salary."""
    cur.execute("UPDATE staff SET salary = ? WHERE name = ?", (new_salary, old_name))
    conn.commit()

def update_employee_position(new_pos: str, old_name: str) -> None:
    """Update an employee's position."""
    cur.execute("UPDATE staff SET position = ? WHERE name = ?", (new_pos, old_name))
    conn.commit()

def get_detail(acc_no: int) -> tuple[str, int] | None:
    """Retrieve the name and balance of a customer account."""
    cur.execute("SELECT name, balance FROM bank WHERE acc_no = ?", (acc_no,))
    return cur.fetchone()

def check_name_in_staff(name: str) -> bool:
    """Check if an employee with the given name exists."""
    cur.execute("SELECT 1 FROM staff WHERE name = ?", (name,))
    return cur.fetchone() is not None

def get_employee_data(name: str) -> tuple[str, str, int, str] | None:
    """Retrieve complete details of a specific employee."""
    cur.execute("SELECT * FROM staff WHERE name = ?", (name,))
    return cur.fetchone()