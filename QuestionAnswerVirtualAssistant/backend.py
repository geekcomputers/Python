import sqlite3
import json
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

class QuestionAnswerVirtualAssistant:
    """
    Used for automatic question-answering

    It works by building a reverse index store that maps
    words to an id. To find the indexed questions that contain
    a certain the words in the user question, we then take an 
    intersection of the ids, ranks the questions to pick the best fit,
    then select the answer that maps to that question
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
        self.conn = sqlite3.connect("virtualassistant.sqlite3", autocommit=True)
        cur = self.conn.cursor()
        res = cur.execute("SELECT name FROM sqlite_master WHERE name='IdToQuesAns'")
        tables_exist = res.fetchone()

        if not tables_exist:
            self.conn.execute("CREATE TABLE IdToQuesAns(id INTEGER PRIMARY KEY, question TEXT, answer TEXT)")
            self.conn.execute('CREATE TABLE WordToId (name TEXT, value TEXT)')
            cur.execute("INSERT INTO WordToId VALUES (?, ?)", ("index", "{}",))

    def index_question_answer(self, question, answer):
        """
        Returns - string
        Input - str: a string of words called question
        ----------
        Indexes the question and answer. It does this by performing two
        operations - add the question and answer to the IdToQuesAns, then
        adds the words in the question to WordToId
        - takes in the question and answer (str)
        - passes the question and answer to a method to add them
          to IdToQuesAns
        - retrieves the id of the inserted ques-answer
        - uses the id to call the method that adds the words of 
          the question to the reverse index WordToId if the word has not
          already been indexed
        """
        row_id = self._add_to_IdToQuesAns(question.lower(), answer.lower())
        cur = self.conn.cursor()
        reverse_idx = cur.execute("SELECT value FROM WordToId WHERE name='index'").fetchone()[0]
        reverse_idx = json.loads(reverse_idx)
        question = question.split()
        for word in question:
            if word not in reverse_idx:
                reverse_idx[word] = [row_id]
            else:
                if row_id not in reverse_idx[word]:
                    reverse_idx[word].append(row_id)
        reverse_idx = json.dumps(reverse_idx)
        cur = self.conn.cursor()
        result = cur.execute("UPDATE WordToId SET value = (?) WHERE name='index'", (reverse_idx,))
        return("index successful")

    def _add_to_IdToQuesAns(self, question, answer):
        """
        Returns - int: the id of the inserted document
        Input - str: a string of words called `document`
        ---------
        - use the class-level connection object to insert the document
          into the db
        - retrieve and return the row id of the inserted document
        """
        cur = self.conn.cursor()
        res = cur.execute("INSERT INTO IdToQuesAns (question, answer) VALUES (?, ?)", (question, answer,))
        return res.lastrowid

    def find_questions(self, user_input):
        """
        Returns - <class method>: the return value of the _find_questions_with_idx method
        Input - str: a string of words called `user_input`, expected to be a question
        ---------
        - retrieve the reverse index
        - use the words contained in the user input to find all the idxs
          that contain the word
        - use idxs to call the _find_questions_with_idx method
        - return the result of the called method
        """
        cur = self.conn.cursor()
        reverse_idx = cur.execute("SELECT value FROM WordToId WHERE name='index'").fetchone()[0]
        reverse_idx = json.loads(reverse_idx)
        user_input = user_input.split(" ")
        all_docs_with_user_input = []
        for term in user_input:
            if term in reverse_idx:
                all_docs_with_user_input.append(reverse_idx[term])

        if not all_docs_with_user_input: # the user_input does not exist
            return []

        common_idx_of_docs = set(all_docs_with_user_input[0])
        for idx in all_docs_with_user_input[1:]:
            common_idx_of_docs.intersection_update(idx)

        if not common_idx_of_docs: # the user_input does not exist
            return []

        return self._find_questions_with_idx(common_idx_of_docs)
        
    def _find_questions_with_idx(self, idxs):
        """
        Returns - list[str]: the list of questions with the idxs
        Input - list of idxs
        ---------
        - use the class-level connection object to retrieve the questions that
          have the idx in the input list of idxs.
        - retrieve and return these questions as a list
        """
        idxs = list(idxs)
        cur = self.conn.cursor()
        sql="SELECT id, question, answer FROM IdToQuesAns WHERE id in ({seq})".format(
                                                                seq=','.join(['?']*len(idxs))
                                                               )
        result = cur.execute(sql, idxs).fetchall()
        return(result)

    def find_most_matched_question(self, user_input, corpus):
        """
        Returns - list[str]: the list of [(score, most_matching_question)]
        Input - user_input, and list of matching questions called corpus
        ---------
        - use the tfidf score to rank the questions and pick the most matching
            question
        """
        vectorizer = TfidfVectorizer()
        tfidf_scores = vectorizer.fit_transform(corpus)
        tfidf_array = pd.DataFrame(tfidf_scores.toarray(),columns=vectorizer.get_feature_names_out())
        tfidf_dict = tfidf_array.to_dict()

        user_input = user_input.split(" ")
        result = []
        for idx in range(len(corpus)):
            result.append([0, corpus[idx]])
            
        for term in user_input:
            if term in tfidf_dict:
                for idx in range(len(result)):
                    result[idx][0] += tfidf_dict[term][idx]
        return result[0]

    def provide_answer(self, user_input):
        """
        Returns - str: the answer to the user_input
        Input - str: user_input
        ---------
        - use the user_input to get the list of matching questions
        - create a corpus which is a list of all matching questions
        - create a question_map that maps questions to their respective answers
        - use the user_input and corpus to find the most matching question
        - return the answer that matches that question from the question_map
        """
        matching_questions = self.find_questions(user_input)
        corpus = [item[1] for item in matching_questions]
        question_map = {question:answer for (id, question, answer) in matching_questions}
        score, most_matching_question = self.find_most_matched_question(user_input, corpus)
        return question_map[most_matching_question]


if __name__ == "__main__":
    va = QuestionAnswerVirtualAssistant()
    va.index_question_answer(
        "What are the different types of competitions available on Kaggle",
        "Types of Competitions Kaggle Competitions are designed to provide challenges for competitors"
    )
    print(
        va.index_question_answer(
            "How to form, manage, and disband teams in a competition",
            "Everyone that competes in a Competition does so as a team. A team is a group of one or more users"
        )
    )
    va.index_question_answer(
        "What is Data Leakage",
        "Data Leakage is the presence of unexpected additional information in the training data"
    )
    va.index_question_answer(
        "How does Kaggle handle cheating",
        "Cheating is not taken lightly on Kaggle. We monitor our compliance account"
    )
    print(va.provide_answer("state Kaggle cheating policy"))
    print(va.provide_answer("Tell me what is data leakage"))