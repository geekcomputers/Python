my_string=input("Enter a string to count number of consonants: ")
string_check=["a", "e", "i", "o", "u", "A", "E", "I", "O", "U"]

def count_con(string):
  c=0
  for i in range(len(string)):
    if string[i] not in string_check:
      c += 1
  return c   

counter = count_con(my_string)
print(f"Number of consonants in {my_string} is {counter}.")
