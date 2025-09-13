import sqlite3
import os


class DatabaseManager:
    def __init__(self, db_name="bankmanaging.db"):
        self.db_path = os.path.join(os.path.dirname(__file__), db_name)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cur = self.conn.cursor()
        self._setup_tables()
        self.acc_no = self._get_last_acc_no() + 1

    def _setup_tables(self):
        self.cur.execute("""
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
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS staff (
                name TEXT,
                pass TEXT,
                salary INTEGER,
                position TEXT
            )
        """)
        self.cur.execute("CREATE TABLE IF NOT EXISTS admin (name TEXT, pass TEXT)")
        self.cur.execute("SELECT COUNT(*) FROM admin")
        if self.cur.fetchone()[0] == 0:
            self.cur.execute("INSERT INTO admin VALUES (?, ?)", ("admin", "admin123"))
        self.conn.commit()

    def _get_last_acc_no(self):
        self.cur.execute("SELECT MAX(acc_no) FROM bank")
        last = self.cur.fetchone()[0]
        return last if last else 0

    # ----------------- Admin -----------------
    def check_admin(self, name, password):
        self.cur.execute("SELECT 1 FROM admin WHERE name=? AND pass=?", (name, password))
        return self.cur.fetchone() is not None

    # ----------------- Staff -----------------
    def create_employee(self, name, password, salary, position):
        self.cur.execute("INSERT INTO staff VALUES (?, ?, ?, ?)", (name, password, salary, position))
        self.conn.commit()

    def check_employee(self, name, password):
        self.cur.execute("SELECT 1 FROM staff WHERE name=? AND pass=?", (name, password))
        return self.cur.fetchone() is not None

    def show_employees(self):
        self.cur.execute("SELECT name, salary, position FROM staff")
        return self.cur.fetchall()

    def update_employee(self, field, new_value, name):
        if field not in {"name", "pass", "salary", "position"}:
            raise ValueError("Invalid employee field")
        self.cur.execute(f"UPDATE staff SET {field}=? WHERE name=?", (new_value, name))
        self.conn.commit()

    def check_name_in_staff(self, name):
        self.cur.execute("SELECT 1 FROM staff WHERE name=?", (name,))
        return self.cur.fetchone() is not None

    # ----------------- Customer -----------------
    def create_customer(self, name, age, address, balance, acc_type, mobile_number):
        acc_no = self.acc_no
        self.cur.execute(
            "INSERT INTO bank VALUES (?, ?, ?, ?, ?, ?, ?)",
            (acc_no, name, age, address, balance, acc_type, mobile_number)
        )
        self.conn.commit()
        self.acc_no += 1
        return acc_no

    def check_acc_no(self, acc_no):
        self.cur.execute("SELECT 1 FROM bank WHERE acc_no=?", (acc_no,))
        return self.cur.fetchone() is not None

    def get_details(self, acc_no):
        self.cur.execute("SELECT * FROM bank WHERE acc_no=?", (acc_no,))
        return self.cur.fetchone()

    def get_detail(self, acc_no):
        self.cur.execute("SELECT name, balance FROM bank WHERE acc_no=?", (acc_no,))
        return self.cur.fetchone()

    def update_customer(self, field, new_value, acc_no):
        if field not in {"name", "age", "address", "mobile_number", "account_type"}:
            raise ValueError("Invalid customer field")
        self.cur.execute(f"UPDATE bank SET {field}=? WHERE acc_no=?", (new_value, acc_no))
        self.conn.commit()

    def update_balance(self, amount, acc_no):
        self.cur.execute("UPDATE bank SET balance = balance + ? WHERE acc_no=?", (amount, acc_no))
        self.conn.commit()

    def deduct_balance(self, amount, acc_no):
        self.cur.execute("SELECT balance FROM bank WHERE acc_no=?", (acc_no,))
        bal = self.cur.fetchone()
        if bal and bal[0] >= amount:
            self.cur.execute("UPDATE bank SET balance=balance-? WHERE acc_no=?", (amount, acc_no))
            self.conn.commit()
            return True
        return False

    def check_balance(self, acc_no):
        self.cur.execute("SELECT balance FROM bank WHERE acc_no=?", (acc_no,))
        bal = self.cur.fetchone()
        return bal[0] if bal else 0

    def list_all_customers(self):
        self.cur.execute("SELECT * FROM bank")
        return self.cur.fetchall()

    def delete_acc(self, acc_no):
        self.cur.execute("DELETE FROM bank WHERE acc_no=?", (acc_no,))
        self.conn.commit()

    # ----------------- Stats -----------------
    def all_money(self):
        self.cur.execute("SELECT SUM(balance) FROM bank")
        total = self.cur.fetchone()[0]
        return total if total else 0

    # ----------------- Cleanup -----------------
    def close(self):
        self.conn.close()
