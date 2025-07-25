import os
import pickle


def initialize_file_if_not_exists(file_path: str) -> None:
    """
    Check if the file exists. If not, create it with an empty list.

    Args:
        file_path (str): Path to the file.

    Raises:
        ValueError: If file_path is empty.
    """
    if not file_path:
        raise ValueError("File path cannot be empty")

    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            pickle.dump([], f)
        print(f"Created new file: {file_path}")


def search_student_record(file_path: str) -> None:
    """
    Search for a student record by roll number in the binary file.

    Args:
        file_path (str): Path to the binary file containing student records.
    """
    initialize_file_if_not_exists(file_path)

    try:
        with open(file_path, "rb") as f:
            records: list[tuple[int, str, float]] = pickle.load(f)

            if not records:
                print("No records found in the file.")
                return

            roll_to_search = int(input("Enter student roll number to search: "))
            found = False

            for record in records:
                if record[0] == roll_to_search:
                    print("\nRecord found:")
                    print(f"Roll: {record[0]}")
                    print(f"Name: {record[1]}")
                    print(f"Percentage: {record[2]:.1f}%")
                    found = True
                    break

            if not found:
                print(f"Record with roll number {roll_to_search} not found.")

    except ValueError:
        print("Error: Invalid roll number. Please enter an integer.")
    except pickle.UnpicklingError:
        print("Error: File content is corrupted and cannot be read.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    FILE_PATH = r"studrec.dat"  # Update with your actual file path
    search_student_record(FILE_PATH)
