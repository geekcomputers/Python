import os
import pickle


def initialize_file_if_not_exists(file_path: str) -> None:
    """
    Check if file exists. If not, create it with an empty list of records.

    Args:
        file_path (str): Path to the file.

    Raises:
        ValueError: If file_path is empty.
    """
    if not file_path:
        raise ValueError("File path cannot be empty.")

    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            pickle.dump([], f)
        print(f"Created new file: {file_path}")


def update_student_record(file_path: str) -> None:
    """
    Update a student's name and marks in the binary file by roll number.

    Args:
        file_path (str): Path to the binary file containing student records.
    """
    initialize_file_if_not_exists(file_path)

    try:
        with open(file_path, "rb+") as f:
            # Load existing records
            records: list[tuple[int, str, int]] = pickle.load(f)

            if not records:
                print("No records found in the file.")
                return

            # Get roll number to update
            roll_to_update = int(input("Enter roll number to update: "))
            found = False

            # Find and update the record
            for i, record in enumerate(records):
                if record[0] == roll_to_update:
                    current_name = record[1]
                    current_marks = record[2]

                    print("\nCurrent Record:")
                    print(f"Roll: {roll_to_update}")
                    print(f"Name: {current_name}")
                    print(f"Marks: {current_marks}")

                    new_name = input(
                        "Enter new name (leave blank to keep current): "
                    ).strip()
                    new_marks_input = input(
                        "Enter new marks (leave blank to keep current): "
                    ).strip()

                    # Update name if provided
                    if new_name:
                        records[i] = (record[0], new_name, record[2])

                    # Update marks if provided
                    if new_marks_input:
                        try:
                            new_marks = int(new_marks_input)
                            if records[i][1] == new_name:  # If name was updated
                                records[i] = (record[0], new_name, new_marks)
                            else:  # If name was not updated
                                records[i] = (record[0], record[1], new_marks)
                        except ValueError:
                            print("Invalid marks input. Marks not updated.")

                    print("Record updated successfully.")
                    found = True
                    break

            if not found:
                print(f"Record with roll number {roll_to_update} not found.")
                return

            # Rewrite the entire file with updated records
            f.seek(0)
            pickle.dump(records, f)
            f.truncate()  # Ensure any remaining data is removed

            # Display updated record
            f.seek(0)
            updated_records = pickle.load(f)
            print("\nUpdated Records:")
            print(f"{'ROLL':<8}{'NAME':<15}{'MARKS':<8}")
            print("-" * 35)
            for record in updated_records:
                print(f"{record[0]:<8}{record[1]:<15}{record[2]:<8}")

    except ValueError:
        print("Error: Invalid roll number. Please enter an integer.")
    except pickle.UnpicklingError:
        print("Error: File content is corrupted and cannot be read.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    FILE_PATH = r"studrec.dat"  # Update with your actual file path
    update_student_record(FILE_PATH)
