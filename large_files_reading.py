with open(
    "new_project.txt", "r", encoding="utf-8"
) as file:  # replace "largefile.text" with your actual file name or with absoulte path
    # encoding = "utf-8" is especially used when the file contains special characters....
    for f in file:
        print(f.strip())
