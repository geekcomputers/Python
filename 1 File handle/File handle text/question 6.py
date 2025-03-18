"""Write a function in python to count the number of lowercase
alphabets present in a text file “happy.txt”"""

from counter import Counter

def lowercase():

    with open("happy.txt") as F:
        word_counter = Counter(F.read())
        
        print(f"The total number of lower case letters are {word_counter.get_total_lower()}")
        print(f"The total number of upper case letters are {word_counter.get_total_upper()}")
        print(f"The total number of letters are {word_counter.get_total()}")

if __name__ == "__main__":
    lowercase()
