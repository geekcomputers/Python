import sqlite3
import json

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

        if not tables_exist:
            self.conn.execute("CREATE TABLE WordMap(name TEXT, value TEXT)")
            self.conn.execute('CREATE TABLE WordPrediction (name TEXT, value TEXT)')
            cur.execute("INSERT INTO WordMap VALUES (?, ?)", ("wordsmap", "{}",))
            cur.execute("INSERT INTO WordPrediction VALUES (?, ?)", ("predictions", "{}",))

    def train(self, sentence):
        """
        Returns - string
        Input - str: a string of words called sentence
        ----------
        Trains the sentence. It does this by creating a map of
        current words to next words and their counts for each
        time the next word appears after the current word
        - takes in the sentence and splits it into a list of words
        - retrieves the word map and predictions map
        - creates the word map and predictions map together
        - saves word map and predictions map to the database
        """
        cur = self.conn.cursor()
        words_list = sentence.split(" ")

        words_map = cur.execute("SELECT value FROM WordMap WHERE name='wordsmap'").fetchone()[0]
        words_map = json.loads(words_map)

        predictions = cur.execute("SELECT value FROM WordPrediction WHERE name='predictions'").fetchone()[0]
        predictions = json.loads(predictions)

        for idx in range(len(words_list)-1):
            curr_word, next_word = words_list[idx], words_list[idx+1]
            if curr_word not in words_map:
                words_map[curr_word] = {}
            if next_word not in words_map[curr_word]:
                words_map[curr_word][next_word] = 1
            else:
                words_map[curr_word][next_word] += 1

            # checking the completion word against the next word
            if curr_word not in predictions:
                predictions[curr_word] = {
                    'completion_word': next_word,
                    'completion_count': 1
                }
            else:
                if words_map[curr_word][next_word] > predictions[curr_word]['completion_count']:
                    predictions[curr_word]['completion_word'] = next_word
                    predictions[curr_word]['completion_count'] = words_map[curr_word][next_word]

        words_map = json.dumps(words_map)
        predictions = json.dumps(predictions)

        cur.execute("UPDATE WordMap SET value = (?) WHERE name='wordsmap'", (words_map,))
        cur.execute("UPDATE WordPrediction SET value = (?) WHERE name='predictions'", (predictions,))
        return("training complete")

    def predict(self, word):
        """
        Returns - string
        Input - string
        ----------
        Returns the completion word of the input word
        - takes in a word
        - retrieves the predictions map
        - returns the completion word of the input word
        """
        cur = self.conn.cursor()
        predictions = cur.execute("SELECT value FROM WordPrediction WHERE name='predictions'").fetchone()[0]
        predictions = json.loads(predictions)
        completion_word = predictions[word.lower()]['completion_word']
        return completion_word



if __name__ == "__main__":
    input_ = "It is not enough to just know how tools work and what they worth,\
              we have got to learn how to use them and to use them well. And with\
              all these new weapons in your arsenal, we would better get those profits fired up"
    ac = AutoComplete()
    ac.train(input_)
    print(ac.predict("to"))