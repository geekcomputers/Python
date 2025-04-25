import datetime

def add_expense(expenses):
    amount = float(input("Enter the expense amount: "))
    category = input("Category (food, travel, shopping, bills, etc.): ")
    date_str = input("Date (YYYY-MM-DD): ")
    try:
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        print("Incorrect date format. Please use YYYY-MM-DD format.")
        return
    note = input("(Optional) Note: ")
    expenses.append({"amount": amount, "category": category, "date": date, "note": note})
    print("Expense added!")

def view_expenses(expenses, period="all", category_filter=None):
    if not expenses:
        print("No expenses recorded yet.")
        return

    filtered_expenses = expenses
    if category_filter:
        filtered_expenses = [e for e in filtered_expenses if e["category"] == category_filter]

    if period == "day":
        date_str = input("Enter the date to view expenses for (YYYY-MM-DD): ")
        try:
            date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            filtered_expenses = [e for e in filtered_expenses if e["date"] == date]
        except ValueError:
            print("Incorrect date format.")
            return
    elif period == "week":
        date_str = input("Enter the start date of the week (YYYY-MM-DD - first day of the week): ")
        try:
            start_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            end_date = start_date + datetime.timedelta(days=6)
            filtered_expenses = [e for e in filtered_expenses if start_date <= e["date"] <= end_date]
        except ValueError:
            print("Incorrect date format.")
            return
    elif period == "month":
        year = input("Enter the year for the month (YYYY): ")
        month = input("Enter the month (MM): ")
        try:
            year = int(year)
            month = int(month)
            filtered_expenses = [e for e in filtered_expenses if e["date"].year == year and e["date"].month == month]
        except ValueError:
            print("Incorrect year or month format.")
            return

    if not filtered_expenses:
        print("No expenses found for this period or category.")
        return

    print("\n--- Expenses ---")
    total_spent = 0
    for expense in filtered_expenses:
        print(f"Amount: {expense['amount']}, Category: {expense['category']}, Date: {expense['date']}, Note: {expense['note']}")
        total_spent += expense['amount']
    print(f"\nTotal spent: {total_spent}")

def save_expenses(expenses, filename="expenses.txt"):
    with open(filename, "w") as f:
        for expense in expenses:
            f.write(f"{expense['amount']},{expense['category']},{expense['date']},{expense['note']}\n")
    print("Expenses saved!")

def load_expenses(filename="expenses.txt"):
    expenses = []
    try:
        with open(filename, "r") as f:
            for line in f:
                amount, category, date_str, note = line.strip().split(',')
                expenses.append({"amount": float(amount), "category": category, "date": datetime.datetime.strptime(date_str, "%Y-%m-%d").date(), "note": note})
    except FileNotFoundError:
        pass
    return expenses

def main():
    expenses = load_expenses()

    while True:
        print("\n--- Personal Expense Tracker ---")
        print("1. Add new expense")
        print("2. View expenses")
        print("3. Filter by category")
        print("4. Save expenses")
        print("5. Exit")

        choice = input("Choose your option: ")

        if choice == '1':
            add_expense(expenses)
        elif choice == '2':
            period = input("View expenses by (day/week/month/all): ").lower()
            view_expenses(expenses, period)
        elif choice == '3':
            category_filter = input("Enter the category to filter by: ")
            view_expenses(expenses, category_filter=category_filter)
        elif choice == '4':
            save_expenses(expenses)
        elif choice == '5':
            save_expenses(expenses)
            print("Thank you!")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()