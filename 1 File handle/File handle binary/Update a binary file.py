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
    Update a student's name in the binary file by roll number.
    
    Args:
        file_path (str): Path to the binary file containing student records.
    """
    initialize_file_if_not_exists(file_path)
    
    try:
        with open(file_path, "rb+") as f:
            # Load existing records
            records: list[tuple[int, str, float]] = pickle.load(f)
            
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
                    new_name = input(f"Current name: {current_name}. Enter new name: ").strip()
                    
                    if new_name:
                        # Create a new tuple with updated name
                        updated_record = (record[0], new_name, record[2])
                        records[i] = updated_record
                        print("Record updated successfully.")
                    else:
                        print("Name cannot be empty. Update cancelled.")
                    
                    found = True
                    break
            
            if not found:
                print(f"Record with roll number {roll_to_update} not found.")
                return
                
            # Rewrite the entire file with updated records
            f.seek(0)
            pickle.dump(records, f)
            f.truncate()  # Ensure any remaining data is removed
            
    except ValueError:
        print("Error: Invalid roll number. Please enter an integer.")
    except pickle.UnpicklingError:
        print("Error: File content is corrupted and cannot be read.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

def display_all_records(file_path: str) -> None:
    """
    Display all student records in the binary file.
    
    Args:
        file_path (str): Path to the binary file.
    """
    initialize_file_if_not_exists(file_path)
    
    try:
        with open(file_path, "rb") as f:
            records: list[tuple[int, str, float]] = pickle.load(f)
            
            if not records:
                print("No records found in the file.")
                return
                
            print("\nAll Student Records:")
            print(f"{'ROLL':<8}{'NAME':<20}{'PERCENTAGE':<12}")
            print("-" * 40)
            
            for record in records:
                print(f"{record[0]:<8}{record[1]:<20}{record[2]:<12.1f}")
                
    except pickle.UnpicklingError:
        print("Error: File content is corrupted and cannot be read.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    FILE_PATH = r"class.dat"  # Update with your actual file path
    
    update_student_record(FILE_PATH)
    display_all_records(FILE_PATH)