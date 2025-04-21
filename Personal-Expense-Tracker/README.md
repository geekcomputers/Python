# Personal Expense Tracker CLI

This is a basic command-line interface (CLI) application built with Python to help you track your daily expenses. It allows you to easily add your expenditures, categorize them, and view your spending patterns over different time periods.

## Features

* **Add New Expense:** Record new expenses by providing the amount, category (e.g., food, travel, shopping, bills), date, and an optional note.
* **View Expenses:** Display your expenses for a specific day, week, month, or all recorded expenses.
* **Filter by Category:** View expenses belonging to a particular category.
* **Data Persistence:** Your expense data is saved to a plain text file (`expenses.txt`) so it's retained between sessions.
* **Simple Command-Line Interface:** Easy-to-use text-based menu for interacting with the application.

## Technologies Used

* **Python:** The core programming language used for the application logic.
* **File Handling:** Used to store and retrieve expense data from a text file.
* **`datetime` module:** For handling and managing date information for expenses.

## How to Run

1.  Make sure you have Python installed on your system.
2.  Save the `expense_tracker.py` file to your local machine.
3.  Open your terminal or command prompt.
4.  Navigate to the directory where you saved the file using the `cd` command.
5.  Run the application by executing the command: `python expense_tracker.py`

## Basic Usage

1.  Run the script. You will see a menu with different options.
2.  To add a new expense, choose option `1` and follow the prompts to enter the required information.
3.  To view expenses, choose option `2` and select the desired time period (day, week, month, or all).
4.  To filter expenses by category, choose option `3` and enter the category you want to view.
5.  To save any new expenses (though the application automatically saves on exit as well), choose option `4`.
6.  To exit the application, choose option `5`.

## Potential Future Enhancements (Ideas for Expansion)

* Implement a monthly budget feature with alerts.
* Add a login system for multiple users.
* Generate visual reports like pie charts for category-wise spending (using libraries like `matplotlib`).
* Incorporate voice input for adding expenses (using `speech_recognition`).
* Migrate data storage to a more structured database like SQLite.

* Add functionality to export expense data to CSV files.

---

> This simple Personal Expense Tracker provides a basic yet functional way to manage your finances from the command line.

#### Author: Dhrubaraj Pati