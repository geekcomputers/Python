import pickle


def delete_student_record() -> None:
    """
    Delete a student record from the binary data file 'studrec.dat' based on the provided roll number.

    This function performs the following operations:
    1. Reads the current student records from 'studrec.dat'
    2. Prompts the user to enter a roll number to delete
    3. Removes the record with the specified roll number
    4. Writes the updated records back to 'studrec.dat'

    Each student record is stored as a tuple in the format: (roll_number, ...)

    Raises:
        FileNotFoundError: If 'studrec.dat' does not exist.
        pickle.UnpicklingError: If the file contains corrupted data.
        ValueError: If the user input cannot be converted to an integer.
    """
    try:
        # Read existing student records from file
        with open("studrec.dat", "rb") as file:
            student_records: list[tuple[int, ...]] = pickle.load(file)
            print("Current student records:", student_records)

        # Get roll number to delete
        roll_number: int = int(input("Enter the roll number to delete: "))

        # Filter out the record with the specified roll number
        updated_records: list[tuple[int, ...]] = [
            record for record in student_records if record[0] != roll_number
        ]

        # Write updated records back to file
        with open("studrec.dat", "wb") as file:
            pickle.dump(updated_records, file)

        print(f"Record with roll number {roll_number} has been deleted.")

    except FileNotFoundError:
        print("Error: The file 'studrec.dat' does not exist.")
    except pickle.UnpicklingError:
        print("Error: The file contains corrupted data.")
    except ValueError as e:
        print(f"Error: Invalid input. Please enter an integer. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    delete_student_record()
