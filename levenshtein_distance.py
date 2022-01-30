def levenshtein_dis(wordA, wordB):

    wordA = wordA.lower()  # making the wordA lower case
    wordB = wordB.lower()  # making the wordB lower case

    # get the length of the words and defining the variables
    length_A = len(wordA)
    length_B = len(wordB)
    max_len = 0
    diff = 0
    distances = []
    distance = 0

    # check the difference of the word to decide how many letter should be delete or add
    # also store that value in the 'diff' variable and get the max length of the user given words
    if length_A > length_B:
        diff = length_A - length_B
        max_len = length_A
    elif length_A < length_B:
        diff = length_B - length_A
        max_len = length_B
    else:
        diff = 0
        max_len = length_A

    # starting from the front of the words and compare the letters of the both user given words
    for x in range(max_len - diff):
        if wordA[x] != wordB[x]:
            distance += 1

    # add the 'distance' value to the 'distances' array
    distances.append(distance)
    distance = 0

    # starting from the back of the words and compare the letters of the both user given words
    for x in range(max_len - diff):
        if wordA[-(x + 1)] != wordB[-(x + 1)]:
            distance += 1

    # add the 'distance' value to the 'distances' array
    distances.append(distance)

    # get the minimun value of the 'distances' array and add it with the 'diff' values and
    # store them in the 'diff' variable
    diff = diff + min(distances)

    # return the value
    return diff
