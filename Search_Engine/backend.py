import sqlite3
import test_data

class SearchEngine:
    """
    It works by building a reverse index store that maps
    words to an id. To find the document(s) that contain
    a certain search term, we then take an intersection
    of the ids
    """

    def __init__(self):
        """
        Return - None
        Input - None
        ----------
        - Initialize database. we use sqlite3
        - Check if the tables exist, if not create them
        - maintain a class level access to the database
          connection object
        """
        self.conn = sqlite3.connect("searchengine.db")
        cur = self.conn.cursor()
        res = cur.execute("SELECT name FROM sqlite_master WHERE name='IdToDoc'")
        tables_exist = res.fetchone()
        # tables_exist = res.fetchall()
        if not tables_exist:
            self.conn.execute("CREATE TABLE IdToDoc(id INTEGER PRIMARY KEY, document TEXT)")
            self.conn.execute('CREATE TABLE WordToId (store TEXT)')
            # self.conn.commit()

        # cur.execute("INSERT INTO DocumentStore (document) VALUES (?)", (document1,))
        # self.conn.commit()
        res = cur.execute("SELECT name FROM sqlite_master")
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
        self._add_to_IdToDoc(document)
        # self._add_to_WordToId(document)
        # doc_num = 1
        # for word in document:
        #     if word not in self.index:
        #         self.index[word] = set([doc_num])
        #     else:
        #         self.index.add(doc_num)
        # print(self.index)

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
    se.index_document("we should all strive to be happy")
    se.index_document("happiness is all you need")
    se.index_document("no way should we be sad")