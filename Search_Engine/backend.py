import sqlite3
import test_data

class SearchEngine:
    def __init__(self, document1):
        # initialize database
        conn = sqlite3.connect("searchengine.db")
        cur = conn.cursor()
        cur.execute("CREATE TABLE DocumentStore(id INTEGER PRIMARY KEY, document TEXT)")
        cur.execute("INSERT INTO DocumentStore (document) VALUES (?)", (document1,))
        conn.commit()
        res = cur.execute("SELECT * FROM DocumentStore")
        print(res.fetchall())
        # self.index = test_data['documents'][:-1]
        # cur = conn.execute('CREATE TABLE keyvals (key TEXT PRIMARY KEY, value TEXT)')

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