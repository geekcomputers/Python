import sqlite3
import test_data

class SearchEngine:
    def __init__(self, document1):
        """
        - Initialize database.
        - Check if the tables exist, if not create them
        """
        self.conn = sqlite3.connect("searchengine.db")
        cur = self.conn.cursor()
        res = cur.execute("SELECT name FROM sqlite_master WHERE name='IndexToWord'")
        tables_exist = res.fetchone()
        # tables_exist = res.fetchall()
        if not tables_exist:
            self.conn.execute("CREATE TABLE IndexToWord(id INTEGER PRIMARY KEY, document TEXT)")
            self.conn.execute('CREATE TABLE WordToIndex (store TEXT)')
            # self.conn.commit()

        # cur.execute("INSERT INTO DocumentStore (document) VALUES (?)", (document1,))
        # self.conn.commit()
        res = cur.execute("SELECT name FROM sqlite_master")
        print(res.fetchall())
        # self.index = test_data['documents'][:-1]
        # 

    def index_document(self, document):
        doc_num = 1
        for word in document:
            if word not in self.index:
                self.index[word] = set([doc_num])
            else:
                self.index.add(doc_num)
        print(self.index)


    def find_documents(self, search_term):
        pass

    def _search_index(self):
        pass

if __name__ == "__main__":
    SearchEngine("we should all strive to be happy")