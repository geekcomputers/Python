import os
import pickle


def read_binary_file() -> None:
    """
    Read student records from a binary file.
    Automatically creates the file with empty records if it doesn't exist.

    Raises:
        pickle.UnpicklingError: If file content is corrupted.
        PermissionError: If unable to create or read the file.
    """
    file_path = r"1 File handle\File handle binary\studrec.dat"

    # Ensure directory exists
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

    # Create empty file if it doesn't exist
    if not os.path.exists(file_path):
        with open(file_path, "wb") as file:
            pickle.dump([], file)  # Initialize with empty list
        print(f"Created new file: {file_path}")

    try:
        # Read student records
        with open(file_path, "rb") as file:
            student_records: list[tuple[int, str, float]] = pickle.load(file)

            # Print records in a formatted table
            print("\nStudent Records:")
            print(f"{'ROLL':<10}{'NAME':<20}{'MARK':<10}")
            print("-" * 40)
            for record in student_records:
                roll, name, mark = record
                print(f"{roll:<10}{name:<20}{mark:<10.1f}")

    except pickle.UnpicklingError:
        print(f"ERROR: File {file_path} is corrupted.")
    except Exception as e:
        print(f"ERROR: Unexpected error - {str(e)}")


if __name__ == "__main__":
    read_binary_file()
