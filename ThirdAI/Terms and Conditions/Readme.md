# ThirdAIApp and NeuralDBClient

This repository contains two components: `ThirdAIApp` and `NeuralDBClient`. `ThirdAIApp` is a graphical user interface (GUI) application for interacting with the ThirdAI neural database client. It allows you to perform training with PDF files and query the database. `NeuralDBClient` is a Python class that serves as a client for interacting with the ThirdAI neural database. It allows you to train the database with PDF files and perform queries to retrieve information.

## ThirdAIApp

### Features

- Insert PDF files for training.
- Train the neural database client.
- Enter queries to retrieve information from the database.
- Display the output in a new window.

### Installation

To run `ThirdAIApp`, you need to have Python and Tkinter installed. You also need the `ThirdAI` library, which you can install using pip:

```bash
pip install ThirdAI
```

### Usage

1. Run the `ThirdAIApp.py` script.
2. The main window will appear.
3. Click the "Insert File!" button to select a PDF file for training.
4. Click the "Training" button to train the neural database client with the selected file.
5. Enter your query in the "Query" field.
6. Click the "Processing" button to process the query and display the output in a new window.
7. You can click the "Clear" button to clear the query and file selections.

### Dependencies

- Python 3.x
- Tkinter
- ThirdAI

## NeuralDBClient

### Features

- Train the neural database with PDF files.
- Perform queries on the neural database.

### Installation

To use `NeuralDBClient`, you need to have the `thirdai` library installed, and you'll need an API key from ThirdAI.

You can install the `thirdai` library using pip:

```bash
pip install thirdai
```

### Usage

1. Import the `NeuralDBClient` class from `neural_db_client.py`.
2. Create an instance of the `NeuralDBClient` class, providing your ThirdAI API key as an argument.

   ```python
   from neural_db_client import NeuralDBClient

   client = NeuralDBClient(api_key="YOUR_API_KEY")
   ```

3. Train the neural database with PDF files using the `train` method. Provide a list of file paths to the PDF files you want to use for training.

   ```python
   client.train(file_paths=["file1.pdf", "file2.pdf"])
   ```

4. Perform queries on the neural database using the `query` method. Provide your query as a string, and the method will return the query results as a string.

   ```python
   result = client.query(question="What is the capital of France?")
   ```

### Dependencies

- `thirdai` library

