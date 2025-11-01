import tkinter as tk
from tkinter import messagebox, simpledialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import pyperclip
import json
from random import choice, randint, shuffle

# ---------------------------- CONSTANTS ------------------------------- #
FONT_NAME = "Helvetica"
# IMP: this is not a secure way to store a master password.
# in a real application, this should be changed and stored securely (e.g., hashed and salted).
MASTER_PASSWORD = "password123"

# ---------------------------- PASSWORD GENERATOR ------------------------------- #
def generate_password():
    """generates a random strong password and copies it to clipboard."""
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    symbols = ['!', '#', '$', '%', '&', '(', ')', '*', '+']

    password_letters = [choice(letters) for _ in range(randint(8, 10))]
    password_symbols = [choice(symbols) for _ in range(randint(2, 4))]
    password_numbers = [choice(numbers) for _ in range(randint(2, 4))]

    password_list = password_letters + password_symbols + password_numbers
    shuffle(password_list)

    password = "".join(password_list)
    password_entry.delete(0, tk.END)
    password_entry.insert(0, password)
    pyperclip.copy(password)
    messagebox.showinfo(title="Password Generated", message="Password copied to clipboard!")

# ---------------------------- SAVE PASSWORD ------------------------------- #
def save():
    """saves the website, email, and password to a JSON file."""
    website = website_entry.get()
    email = email_entry.get()
    password = password_entry.get()
    new_data = {
        website: {
            "email": email,
            "password": password,
        }
    }

    if not website or not password:
        messagebox.showerror(title="Oops", message="Please don't leave any fields empty!")
        return

    is_ok = messagebox.askokcancel(title=website, message=f"These are the details entered: \nEmail: {email} "
                                                      f"\nPassword: {password} \nIs it ok to save?")
    if is_ok:
        try:
            with open("data.json", "r") as data_file:
                data = json.load(data_file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        
        data.update(new_data)

        with open("data.json", "w") as data_file:
            json.dump(data, data_file, indent=4)

        website_entry.delete(0, tk.END)
        password_entry.delete(0, tk.END)

# ---------------------------- FIND PASSWORD ------------------------------- #
def find_password():
    """finds and displays password for a given website."""
    website = website_entry.get()
    try:
        with open("data.json", "r") as data_file:
            data = json.load(data_file)
    except (FileNotFoundError, json.JSONDecodeError):
        messagebox.showerror(title="Error", message="No Data File Found.")
        return
    
    if website in data:
        email = data[website]["email"]
        password = data[website]["password"]
        messagebox.showinfo(title=website, message=f"Email: {email}\nPassword: {password}")
        pyperclip.copy(password)
        messagebox.showinfo(title="Copied", message="Password for {} copied to clipboard.".format(website))
    else:
        messagebox.showerror(title="Error", message=f"No details for {website} exists.")

# ---------------------------- VIEW ALL PASSWORDS ------------------------------- #
def view_all_passwords():
    """prompts for master password and displays all saved passwords if correct."""
    password = simpledialog.askstring("Master Password", "Please enter the master password:", show='*')
    
    if password == MASTER_PASSWORD:
        show_passwords_window()
    elif password is not None: # avoids error message if user clicks cancel
        messagebox.showerror("Incorrect Password", "The master password you entered is incorrect.")

def show_passwords_window():
    """creates a new window to display all passwords in a table."""
    all_passwords_window = tk.Toplevel(window)
    all_passwords_window.title("All Saved Passwords")
    all_passwords_window.config(padx=20, pady=20)
    
    # a frame for the treeview and scrollbar
    tree_frame = ttk.Frame(all_passwords_window)
    tree_frame.grid(row=0, column=0, columnspan=2, sticky='nsew')
    
    # a Treeview (table)
    cols = ('Website', 'Email', 'Password')
    tree = ttk.Treeview(tree_frame, columns=cols, show='headings')
    
    # column headings and widths
    tree.heading('Website', text='Website')
    tree.column('Website', width=150)
    tree.heading('Email', text='Email/Username')
    tree.column('Email', width=200)
    tree.heading('Password', text='Password')
    tree.column('Password', width=200)
    
    tree.grid(row=0, column=0, sticky='nsew')

    # a scrollbar
    scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.grid(row=0, column=1, sticky='ns')

    # load data from JSON file
    try:
        with open("data.json", "r") as data_file:
            data = json.load(data_file)
        
        # insert data into the treeview
        for website, details in data.items():
            tree.insert("", "end", values=(website, details['email'], details['password']))
            
    except (FileNotFoundError, json.JSONDecodeError):
        # if file not found or empty, it will just show an empty table
        pass
    
    def copy_selected_info(column_index, info_type):
        """copies the email or password of the selected row."""
        selected_item = tree.focus()
        if not selected_item:
            messagebox.showwarning("No Selection", "Please select a row from the table first.", parent=all_passwords_window)
            return
            
        item_values = tree.item(selected_item, 'values')
        info_to_copy = item_values[column_index]
        pyperclip.copy(info_to_copy)
        messagebox.showinfo("Copied!", f"The {info_type.lower()} for '{item_values[0]}' has been copied to your clipboard.", parent=all_passwords_window)

    # a frame for the buttons
    button_frame = ttk.Frame(all_passwords_window)
    button_frame.grid(row=1, column=0, columnspan=2, pady=(10,0))

    copy_email_button = ttk.Button(button_frame, text="Copy Email", style="success.TButton", command=lambda: copy_selected_info(1, "Email"))
    copy_email_button.pack(side=tk.LEFT, padx=5)

    copy_password_button = ttk.Button(button_frame, text="Copy Password", style="success.TButton", command=lambda: copy_selected_info(2, "Password"))
    copy_password_button.pack(side=tk.LEFT, padx=5)

    all_passwords_window.grab_set() # makes window modal

# ---------------------------- UI SETUP ------------------------------- #
window = ttk.Window(themename="superhero")
window.title("Password Manager")
window.config(padx=50, pady=50)

# logo
canvas = tk.Canvas(width=200, height=200, highlightthickness=0)
logo_img = tk.PhotoImage(file="logo.png")
canvas.create_image(100, 100, image=logo_img)
canvas.grid(row=0, column=1, pady=(0, 20))

# labels
website_label = ttk.Label(text="Website:", font=(FONT_NAME, 12))
website_label.grid(row=1, column=0, sticky="W")
email_label = ttk.Label(text="Email/Username:", font=(FONT_NAME, 12))
email_label.grid(row=2, column=0, sticky="W")
password_label = ttk.Label(text="Password:", font=(FONT_NAME, 12))
password_label.grid(row=3, column=0, sticky="W")

# entries
website_entry = ttk.Entry(width=32)
website_entry.grid(row=1, column=1, pady=5, sticky="EW")
website_entry.focus()
email_entry = ttk.Entry(width=50)
email_entry.grid(row=2, column=1, columnspan=2, pady=5, sticky="EW")
email_entry.insert(0, "example@email.com")
password_entry = ttk.Entry(width=32)
password_entry.grid(row=3, column=1, pady=5, sticky="EW")

# buttons
search_button = ttk.Button(text="Search", width=14, command=find_password, style="info.TButton")
search_button.grid(row=1, column=2, sticky="EW", padx=(5,0))
generate_password_button = ttk.Button(text="Generate Password", command=generate_password, style="success.TButton")
generate_password_button.grid(row=3, column=2, sticky="EW", padx=(5,0))
add_button = ttk.Button(text="Add", width=43, command=save, style="primary.TButton")
add_button.grid(row=4, column=1, columnspan=2, pady=(10,0), sticky="EW")

view_all_button = ttk.Button(text="View All Passwords", command=view_all_passwords, style="secondary.TButton")
view_all_button.grid(row=5, column=1, columnspan=2, pady=(10,0), sticky="EW")


window.mainloop()

