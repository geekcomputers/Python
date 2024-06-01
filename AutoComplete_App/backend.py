import sqlite3
# import test_data
# import ast
# import json

class AutoComplete:
    """
    It works by building a `WordMap` that stores words to word-follower-count
    ----------------------------
    e.g. To train the following statement:
    
    It is not enough to just know how tools work and what they worth,
    we have got to learn how to use them and to use them well.
    And with all these new weapons in your arsenal, we would better
    get those profits fired up

    we create the following:
    {   It: {is:1}
        is: {not:1}
        not: {enough:1}
        enough: {to:1}
        to: {just:1, learn:1, use:2}
        just: {know:1}
        .
        .
        profits: {fired:1}
        fired: {up:1}
    }
    so the word completion for "to" will be "use".
    For optimization, we use another store `WordPrediction` to save the
    predictions for each word
    """

    def __init__(self):
        """
        Returns - None
        Input - None
        ----------
        - Initialize database. we use sqlite3
        - Check if the tables exist, if not create them
        - maintain a class level access to the database
          connection object
        """
        self.conn = sqlite3.connect("autocompleteDB.sqlite3", autocommit=True)
        cur = self.conn.cursor()
        res = cur.execute("SELECT name FROM sqlite_master WHERE name='WordMap'")
        tables_exist = res.fetchone()
        print(tables_exist)

        if not tables_exist:
            self.conn.execute("CREATE TABLE WordMap(name TEXT, value TEXT)")
            self.conn.execute('CREATE TABLE WordPrediction (name TEXT, value TEXT)')
            cur.execute("INSERT INTO WordMap VALUES (?, ?)", ("wordsmap", "{}",))
            cur.execute("INSERT INTO WordPrediction VALUES (?, ?)", ("predictions", "{}",))

    def train(self, sentence):
        words_list = sentence.split(" ")
        words_map = {}
        for idx in range(len(words_list)-1):
            curr_word, next_word = words_list[idx], words_list[idx+1]
            if curr_word not in words_map:
                words_map[curr_word] = {}
            if next_word not in words_map[curr_word]:
                words_map[curr_word][next_word] = 1
            else:
                words_map[curr_word][next_word] += 1

        print(words_map)


if __name__ == "__main__":
    input_ = "It is not enough to just know how tools work and what they worth,\
              we have got to learn how to use them and to use them well. And with\
              all these new weapons in your arsenal, we would better get those profits fired up"
    ac = AutoComplete()
    print(ac.train(input_))
    # se.index_document("we should all strive to be happy and happy again")
    # print(se.index_document("happiness is all you need"))
    # se.index_document("no way should we be sad")
    # se.index_document("a cheerful heart is a happy one even in Nigeria")
    # print(se.find_documents("happy"))