import sqlite3
import test_data
import ast

class SearchEngine:
    """
    It works by building a reverse index store that maps
    words to an id. To find the document(s) that contain
    a certain search term, we then take an intersection
    of the ids
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
        self.conn = sqlite3.connect("searchengine.db")
        self.cur = self.conn.cursor()
        res = self.cur.execute("SELECT name FROM sqlite_master WHERE name='IdToDoc'")
        tables_exist = res.fetchone()

        if not tables_exist:
            self.conn.execute("CREATE TABLE IdToDoc(id INTEGER PRIMARY KEY, document TEXT)")
            self.conn.execute('CREATE TABLE WordToId (name TEXT, value TEXT)')
            self.cur.execute("INSERT INTO WordToId VALUES (?, ?)", ("index", "{}",))
            # self.conn.commit()

        # cur.execute("INSERT INTO DocumentStore (document) VALUES (?)", (document1,))
        # self.conn.commit()
        res = self.cur.execute("SELECT name FROM sqlite_master")
        print(res.fetchall())
        # self.index = test_data['documents'][:-1]
        # 

    def index_document(self, document):
        """
        Returns - 
        Input - str: a string of words called document
        ----------
        Indexes the document. It does this by performing two
        operations - add the document to the IdToDoc, then
        adds the words in the document to WordToId
        - takes in the document (str)
        - passes the document to a method to add the document
          to IdToDoc
        - retrieves the id of the inserted document
        - uses the id to call the method that adds the words of 
          the document to the index WordToId
        """
        row_id = self._add_to_IdToDoc(document)
        reverse_idx = self.cur.execute("SELECT value FROM WordToId WHERE name='index'").fetchone()[0]
        reverse_idx = ast.literal_eval(reverse_idx)
        document = document.split()
        for word in document:
            if word not in reverse_idx:
                reverse_idx[word] = set([row_id])
            else:
                reverse_idx.add(row_id)
        print(reverse_idx)

    def _add_to_IdToDoc(self, document):
        """
        Returns - int: the id of the inserted document
        Input - str: a string of words called `document`
        ---------
        - use the class-level connection object to insert the document
          into the db
        - retrieve and return the row id of the inserted document
        """
        res = self.conn.execute("INSERT INTO IdToDoc (document) VALUES (?)", (document,))
        return res.lastrowid



    def find_documents(self, search_term):
        pass

    def _search_index(self):
        pass

if __name__ == "__main__":
    se = SearchEngine()
    se.index_document("we should all strive to be happy and happy again")
    se.index_document("happiness is all you need")
    se.index_document("no way should we be sad")