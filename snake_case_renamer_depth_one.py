import os

repository_path = r"D:\coding\Python-Programs"

def rename_files_and_folders(directory):
    for name in os.listdir(directory):
        old_path = os.path.join(directory, name)
        new_name = name.lower().replace(" ", "_")
        new_path = os.path.join(directory, new_name)

        # Check if the new filename is different from the old filename
        if new_name != name:
            # Check if the new filename already exists in the directory
            if os.path.exists(new_path):
                # If the new filename exists, add a number at the end to make it unique
                index = 1
                while os.path.exists(f"{os.path.splitext(new_path)[0]}_{index}{os.path.splitext(new_path)[1]}"):
                    index += 1
                new_path = f"{os.path.splitext(new_path)[0]}_{index}{os.path.splitext(new_path)[1]}"
            
            os.rename(old_path, new_path)

rename_files_and_folders(repository_path)
