import os
import pickle

StudentRecord = tuple[int, str, float]

def initialize_file_if_not_exists(file_path: str) -> None:
    """
    Check if file exists, create and initialize with empty list if not
    """
    if not file_path:
        raise ValueError("File path cannot be empty")
    
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory created: {dir_path}")
    
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            pickle.dump([], f)
        print(f"File initialized: {file_path}")

def write_sample_data(file_path: str) -> None:
    """
    Write sample student data for demonstration
    """
    sample_data: list[StudentRecord] = [
        (1, "Ramya", 30.0),
        (2, "Vaishnavi", 60.0),
        (3, "Anuya", 40.0),
        (4, "Kamala", 30.0),
        (5, "Anuraag", 10.0),
        (6, "Reshi", 77.0),
        (7, "Biancaa.R", 100.0),
        (8, "Sandhya", 65.0),
    ]

    with open(file_path, "wb") as f:
        pickle.dump(sample_data, f)
    print(f"Sample data written to {file_path}")

def count_remedial_students(file_path: str) -> None:
    """
    Count and display students needing remedial classes (percentage < 40)
    """
    initialize_file_if_not_exists(file_path)

    try:
        with open(file_path, "rb") as f:
            students: list[StudentRecord] = pickle.load(f)
            remedial = [s for s in students if s[2] < 40.0]

            print("\nStudents requiring remedial classes:")
            for s in remedial:
                print(f"Roll: {s[0]}, Name: {s[1]}, Percentage: {s[2]:.1f}")
            
            print(f"\nTotal remedial students: {len(remedial)}")

    except pickle.UnpicklingError:
        print(f"Error: File {file_path} is corrupted")
    except Exception as e:
        print(f"Error: {str(e)}")

def count_top_scorers(file_path: str) -> None:
    """
    Count and display students with highest percentage
    """
    initialize_file_if_not_exists(file_path)

    try:
        with open(file_path, "rb") as f:
            students: list[StudentRecord] = pickle.load(f)

            if not students:
                print("No student records found")
                return

            max_percentage = max(s[2] for s in students)
            top_scorers = [s for s in students if s[2] == max_percentage]

            print(f"\nTop scorers with {max_percentage:.1f}%:")
            for s in top_scorers:
                print(f"Roll: {s[0]}, Name: {s[1]}")
            
            print(f"\nTotal top scorers: {len(top_scorers)}")

    except pickle.UnpicklingError:
        print(f"Error: File {file_path} is corrupted")
    except Exception as e:
        print(f"Error: {str(e)}")

def display_all_students(file_path: str) -> None:
    """
    Display all student records in tabular format
    """
    initialize_file_if_not_exists(file_path)

    try:
        with open(file_path, "rb") as f:
            students: list[StudentRecord] = pickle.load(f)

            print("\nAll student records:")
            print(f"{'ROLL':<8}{'NAME':<15}{'PERCENTAGE':<12}")
            print("-" * 35)

            for s in students:
                print(f"{s[0]:<8}{s[1]:<15}{s[2]:<12.1f}")

    except pickle.UnpicklingError:
        print(f"Error: File {file_path} is corrupted")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    FILE_PATH = r"1 File handle\File handle binary\class.dat"

    # Uncomment below line to write sample data (only needed once)
    # write_sample_data(FILE_PATH)

    count_remedial_students(FILE_PATH)
    count_top_scorers(FILE_PATH)
    display_all_students(FILE_PATH)    