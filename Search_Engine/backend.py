import sqlite3
import test_data
import ast
import json

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
        cur = self.conn.cursor()
        res = cur.execute("SELECT name FROM sqlite_master WHERE name='IdToDoc'")
        tables_exist = res.fetchone()

        if not tables_exist:
            self.conn.execute("CREATE TABLE IdToDoc(id INTEGER PRIMARY KEY, document TEXT)")
            self.conn.execute('CREATE TABLE WordToId (name TEXT, value TEXT)')
            cur.execute("INSERT INTO WordToId VALUES (?, ?)", ("index", "{}",))

        cur = self.conn.cursor()
        # res = cur.execute("SELECT name FROM sqlite_master")
        # print(res.fetchall())
        # self.index = test_data['documents'][:-1]
        # 

    def index_document(self, document):
        """
        Returns - <sqlite3.Cursor object>
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
          the document to the reverse index WordToId if the word has not
          already been indexed
        """
        row_id = self._add_to_IdToDoc(document)
        cur = self.conn.cursor()
        reverse_idx = cur.execute("SELECT value FROM WordToId WHERE name='index'").fetchone()[0]
        reverse_idx = json.loads(reverse_idx)
        document = document.split()
        for word in document:
            if word not in reverse_idx:
                reverse_idx[word] = [row_id]
            else:
                if row_id not in reverse_idx[word]:
                    reverse_idx[word].append(row_id)
        reverse_idx = json.dumps(reverse_idx)
        cur = self.conn.cursor()
        result = cur.execute("UPDATE WordToId SET value = (?) WHERE name='index'", (reverse_idx,))
        return(result)

    def _add_to_IdToDoc(self, document):
        """
        Returns - int: the id of the inserted document
        Input - str: a string of words called `document`
        ---------
        - use the class-level connection object to insert the document
          into the db
        - retrieve and return the row id of the inserted document
        """
        cur = self.conn.cursor()
        res = cur.execute("INSERT INTO IdToDoc (document) VALUES (?)", (document,))
        return res.lastrowid

    def find_documents(self, search_term):
        """
        Returns - <class method>: the return value of the _find_documents_with_idx method
        Input - str: a string of words called `search_term`
        ---------
        - retrieve the reverse index
        - use the words contained in the search term to find all the idxs
          that contain the word
        - use idxs to call the _find_documents_with_idx method
        - return the result of the called method
        """
        cur = self.conn.cursor()
        reverse_idx = cur.execute("SELECT value FROM WordToId WHERE name='index'").fetchone()[0]
        reverse_idx = json.loads(reverse_idx)
        search_term = search_term.split(" ")
        all_docs_with_search_term = []
        for term in search_term:
            if term in reverse_idx:
                all_docs_with_search_term.append(reverse_idx[term])

        if not all_docs_with_search_term: # the search term does not exist
            return []

        common_idx_of_docs = set(all_docs_with_search_term[0])
        for idx in all_docs_with_search_term[1:]:
            common_idx_of_docs.intersection_update(idx)

        if not common_idx_of_docs: # the search term does not exist
            return []

        return self._find_documents_with_idx(common_idx_of_docs)
        
    def _find_documents_with_idx(self, idxs):
        idxs = list(idxs)
        cur = self.conn.cursor()
        sql="SELECT document FROM IdToDoc WHERE id in ({seq})".format(
                                                                seq=','.join(['?']*len(idxs))
                                                               )
        result = cur.execute(sql, idxs).fetchall()
        return(result)


if __name__ == "__main__":
    se = SearchEngine()
    se.index_document("we should all strive to be happy and happy again")
    print(se.index_document("happiness is all you need"))
    se.index_document("no way should we be sad")
    se.index_document("a cheerful heart is a happy one")
    print(se.find_documents("happy"))