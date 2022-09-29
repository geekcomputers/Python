def soundex_al(word):
    cap_word = word.upper()  # convert the word to uppercase

    return_val = ""
    return_val = f"{cap_word[0]}"

    # dictonary to give values to the letters
    code_dict = {"BFPV": "1", "CGJKQSXZ": "2", "DT": "3", "L": "4", "MN": "5", "R": "6"}

    # array of charactors to remove from the word
    rem_charactors = ["A", "E", "I", "O", "U", "H", "W", "Y"]

    # for loop to remove all the 0 valued charactors
    temp = ""
    for char in cap_word[1:]:
        if char not in rem_charactors:
            temp = temp + char

    # get the values from the 'code_dict' and create the soundex code
    for char in temp:
        for key, code in code_dict.items():
            if char in key and code != return_val[-1]:
                return_val += code

    return_val = return_val[:4]  # crop the word to 4 charactors

    # if soundex code doen't contain 4 digits. fill it with zeros
    if len(return_val) < 4:
        for _ in range(len(return_val), 4):
            return_val = f"{return_val}0"

    # return the value
    return return_val


# testing the fucntion
print(soundex_al("Danus"))
