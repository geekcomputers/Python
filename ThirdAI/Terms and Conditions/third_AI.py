from thirdai import licensing, neural_db as ndb

licensing.activate("1FB7DD-CAC3EC-832A67-84208D-C4E39E-V3")

db = ndb.NeuralDB(user_id="my_user")
flag = 0


def training(file_path):
    global flag
    insertable_docs = []
    # pdf_files = ["C:/Users/patel/OneDrive/Desktop/XYZ product.pdf"]
    pdf_files = file_path

    for file in pdf_files:
        pdf_doc = ndb.PDF(file)
        insertable_docs.append(pdf_doc)

    print(insertable_docs)

    source_ids = db.insert(insertable_docs, train=True)
    flag += 1
    return flag


def query(question):
    search_results = db.search(
        query=question,
        top_k=2,
        on_error=lambda error_msg: print(f"Error! {error_msg}"))

    for result in search_results:
        print(result.text)
        print('************')
