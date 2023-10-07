from thirdai import licensing, neural_db as ndb


class NeuralDBClient:
    def __init__(self):
        # Activating ThirdAI Key
        licensing.activate("ADD-YOUR-THIRDAI-ACTIVATION-KEY")

        # Creating NeuralBD variable to access Neural Database
        self.db = ndb.NeuralDB(user_id="my_user")

    def train(self, file_paths):
        # Retrieving path of file
        insertable_docs = []
        pdf_files = file_paths

        # Appending PDF file to the Database stack
        pdf_doc = ndb.PDF(pdf_files)
        insertable_docs.append(pdf_doc)

        # Inserting/Uploading PDF file to Neural database for training
        self.db.insert(insertable_docs, train=True)

    def query(self, question):
        # Searching of required query in neural database
        search_results = self.db.search(
            query=question,
            top_k=2,
            on_error=lambda error_msg: print(f"Error! {error_msg}"))

        output = ""
        for result in search_results:
            output += result.text + "\n\n"

        return output

