"""Write a function in python to count the number of lowercase
alphabets present in a text file “happy.txt"""

import time, os
from counter import Counter

print("You will see the count of lowercase, uppercase and total count of alphabets in provided file..")


file_path = input("Please, Enter file path: ")

if os.path.exists(file_path):
    print('The file exists and this is the path:\n',file_path) 


def lowercase(file_path):
    try:

        with open(file_path) as F:
            word_counter = Counter(F.read())
            
            print(f"The total number of lower case letters are {word_counter.get_total_lower()}")
            time.sleep(0.5)
            print(f"The total number of upper case letters are {word_counter.get_total_upper()}")
            time.sleep(0.5)
            print(f"The total number of letters are {word_counter.get_total()}")
            time.sleep(0.5)

    except FileNotFoundError:
        print("File is not exist.. Please check AGAIN")




if __name__ == "__main__":

    lowercase(file_path)
