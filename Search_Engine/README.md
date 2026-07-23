Python Program to search through various documents and return the documents containing the search term. Algorithm involves using a reverse index to store each word in each document where a document is defined by an index. To get the document that contains a search term, we simply find an intersect of all the words in the search term, and using the resulting indexes, retrieve the document(s) that contain these words

To use directly, run

```python3 backend.py```

To use a gui, run

```python3 frontend.py```