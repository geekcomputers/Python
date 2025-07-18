import os
import pickle
from typing import List, Tuple

def initialize_file_if_not_exists(file_path: str) -> None:
    """
    Check if the file exists. If not, create it and initialize with an empty list.
    
    Args:
        file_path (str): Path to the file to check/initialize.
    
    Raises:
        ValueError: If file_path is empty.
    """
    if not file_path:
        raise ValueError("File path cannot be empty.")
    
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created: {directory}")
    
    if not os.path.exists(file_path):
        with open(file_path, "wb") as file:
            pickle.dump([], file)
        print(f"File initialized: {file_path}")

def write_sample_data(file_path: str) -> None:
    """
    Write sample student data to the file for demonstration purposes.
    
    Args:
        file_path (str): Path to the file to write data to.
    """
    sample_data: List[Tuple[int, str, float]] = [
        (1, "Ramya", 30.0),
        (2, "Vaishnavi", 60.0),
        (3, "Anuya", 40.0),
        (4, "Kamala", 30.0),
        (5, "Anuraag", 10.0),
        (6, "Reshi", 77.0),
        (7, "Biancaa.R", 100.0),
        (8, "Sandhya", 65.0),
    ]
    
    with open(file_path, "wb") as file:
        pickle.dump(sample_data, file)
    print(f"Sample data written to {file_path}")

def count_remedial_students(file_path: str) -> None:
    """
    Count and print students who need remedial classes (percentage < 40).
    
    Args:
        file_path (str): Path to the file containing student records.
    """
    initialize_file_if_not_exists(file_path)
    
    try:
        with open(file_path, "rb") as file:
            students: List[Tuple[int, str, float]] = pickle.load(file)
            
            remedial_students = [student for student in students if student[2] < 40.0]
            
            print("\nStudents eligible for remedial classes:")
            for student in remedial_students:
                print(f"Roll: {student[0]}, Name: {student[1]}, Percentage: {student[2]:.1f}")
                
            print(f"\nTotal students needing remedial: {len(remedial_students)}")
            
    except pickle.UnpicklingError:
        print(f"Error: File {file_path} is corrupted.")
    except Exception as e:
        print(f"Error: {str(e)}")

def count_top_scorers(file_path: str) -> None:
    """
    Count and print students who achieved the highest percentage.
    
    Args:
        file_path (str): Path to the file containing student records.
    """
    initialize_file_if_not_exists(file_path)
    
    try:
        with open(file_path, "rb") as file:
            students: List[Tuple[int, str, float]] = pickle.load(file)
            
            if not students:
                print("No student records found.")
                return
                
            max_percentage = max(student[2] for student in students)
            top_scorers = [student for student in students if student[2] == max_percentage]
            
            print(f"\nTop scorers with {max_percentage:.1f}%:")
            for student in top_scorers:
                print(f"Roll: {student[0]}, Name: {student[1]}")
                
            print(f"\nTotal top scorers: {len(top_scorers)}")
            
    except pickle.UnpicklingError:
        print(f"Error: File {file_path} is corrupted.")
    except Exception as e:
        print(f"Error: {str(e)}")

def display_all_students(file_path: str) -> None:
    """
    Display all student records.
    
    Args:
        file_path (str): Path to the file containing student records.
    """
    initialize_file_if_not_exists(file_path)
    
    try:
        with open(file_path, "rb") as file:
            students: List[Tuple[int, str, float]] = pickle.load(file)
            
            print("\nAll student records:")
            print(f"{'ROLL':<8}{'NAME':<15}{'PERCENTAGE':<12}")
            print("-" * 35)
            
            for student in students:
                print(f"{student[0]:<8}{student[1]:<15}{student[2]:<12.1f}")
                
    except pickle.UnpicklingError:
        print(f"Error: File {file_path} is corrupted.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":

    FILE_PATH = r"1 File handle\File handle binary\class.dat"
    
    # Uncomment below line to write sample data (only needed once)
    # write_sample_data(FILE_PATH)
    
    count_remedial_students(FILE_PATH)
    count_top_scorers(FILE_PATH)
    display_all_students(FILE_PATH)