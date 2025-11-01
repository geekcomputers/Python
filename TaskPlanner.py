import csv


def load_tasks(filename="tasks.csv"):
    tasks = []
    with open(filename, "r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            tasks.append({"task": row[0], "deadline": row[1], "completed": row[2]})
    return tasks


def save_tasks(tasks, filename="tasks.csv"):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        for task in tasks:
            writer.writerow([task["task"], task["deadline"], task["completed"]])


def add_task(task, deadline):
    tasks = load_tasks()
    tasks.append({"task": task, "deadline": deadline, "completed": "No"})
    save_tasks(tasks)
    print("Task added successfully!")


def show_tasks():
    tasks = load_tasks()
    for task in tasks:
        print(
            f"Task: {task['task']}, Deadline: {task['deadline']}, Completed: {task['completed']}"
        )


# Example usage
add_task("Write daily report", "2024-04-20")
show_tasks()
