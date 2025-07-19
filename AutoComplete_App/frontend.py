"""
Autocomplete System GUI with N-gram optimization

A graphical user interface for interacting with the autocomplete system.
This interface allows users to train the system with text and get word predictions.
"""

from tkinter import Button, Entry, Label, Tk

from backendgui import AutoComplete


def train() -> None:
    """
    Train the autocomplete system with the text from the training entry field.
    
    Retrieves the input sentence from the training Entry widget,
    creates an instance of AutoComplete, and trains the system with the sentence.
    """
    sentence: str = train_entry.get()
    if sentence.strip(): 
        autocomplete: AutoComplete = AutoComplete(n=2)
        result: str = autocomplete.train(sentence)
        print(result)
    else:
        print("Please enter a non-empty sentence for training.")


def predict_word() -> None:
    """
    Get prediction for the word from the prediction entry field.
    
    Retrieves the input word sequence from the prediction Entry widget,
    uses the autocomplete system to get the most likely next word,
    and prints the result.
    """
    words: str = predict_word_entry.get()
    if words.strip():  
        autocomplete: AutoComplete = AutoComplete(n=2)
        prediction: str | None = autocomplete.predict(words)
        if prediction:
            print(f"Prediction for '{words}': {prediction}")
        else:
            print(f"No prediction available for '{words}'.")
    else:
        print("Please enter a non-empty word sequence for prediction.")


if __name__ == "__main__":
    root: Tk = Tk()
    root.title("Autocomplete System")
    root.geometry('400x300')  
    
    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
    root.grid_rowconfigure(2, weight=1)
    root.grid_rowconfigure(3, weight=1)
    root.grid_columnconfigure(0, weight=1)

    train_label: Label = Label(root, text="Enter text to train:")
    train_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
    
    train_entry: Entry = Entry(root, width=50)
    train_entry.grid(row=1, column=0, padx=10, pady=5, sticky='ew')
    
    train_button: Button = Button(root, text="Train System", command=train)
    train_button.grid(row=2, column=0, padx=10, pady=5, sticky='ew')

    predict_word_label: Label = Label(root, text="Enter word sequence to predict next term:")
    predict_word_label.grid(row=3, column=0, padx=10, pady=5, sticky='w')
    
    predict_word_entry: Entry = Entry(root, width=50)
    predict_word_entry.grid(row=4, column=0, padx=10, pady=5, sticky='ew')
    
    predict_button: Button = Button(root, text="Get Prediction", command=predict_word)
    predict_button.grid(row=5, column=0, padx=10, pady=5, sticky='ew')

    root.mainloop()