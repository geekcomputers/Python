import tkinter as tk
from tkinter.font import Font
from tkinter import messagebox
from tkinter import filedialog
from ThirdAI import NeuralDBClient as Ndb


class ThirdAIApp:
    """
    A GUI application for using the ThirdAI neural database client to train and query data.
    """
    def __init__(self, root):
        """
        Initialize the user interface window.

        Args:
            root (tk.Tk): The main Tkinter window.
        """
        # Initialize the main window
        self.root = root
        self.root.geometry("600x500")
        self.root.title('ThirdAI - T&C')

        # Initialize variables
        self.path = []
        self.client = Ndb()

        # GUI elements

        # Labels and buttons
        self.menu = tk.Label(self.root, text="Terms & Conditions", font=self.custom_font(30), fg='black',
                             highlightthickness=2, highlightbackground="red")
        self.menu.place(x=125, y=10)

        self.insert_button = tk.Button(self.root, text="Insert File!", font=self.custom_font(15), fg='black', bg="grey",
                                       width=10, command=self.file_input)
        self.insert_button.place(x=245, y=100)

        self.text_box = tk.Text(self.root, wrap=tk.WORD, width=30, height=1)
        self.text_box.place(x=165, y=150)

        self.training_button = tk.Button(self.root, text="Training", font=self.custom_font(15), fg='black', bg="grey",
                                         width=10, command=self.training)
        self.training_button.place(x=245, y=195)

        self.query_label = tk.Label(self.root, text="Query", font=self.custom_font(20), fg='black')
        self.query_label.place(x=255, y=255)

        self.query_entry = tk.Entry(self.root, font=self.custom_font(20), width=30)
        self.query_entry.place(x=70, y=300)

        self.processing_button = tk.Button(self.root, text="Processing", font=self.custom_font(15), fg='black',
                                           bg="grey", width=10, command=self.processing)
        self.processing_button.place(x=245, y=355)

        self.clear_button = tk.Button(self.root, text="Clear", font=15, fg='black', bg="grey", width=10,
                                      command=self.clear_all)
        self.clear_button.place(x=245, y=405)

    @staticmethod
    def custom_font(size):
        """
        Create a custom font with the specified size.

        Args:
            size (int): The font size.

        Returns:
            Font: The custom Font object.
        """
        return Font(size=size)

    def file_input(self):
        """
        Open a file dialog to select a PDF file and display its name in the text box.
        """
        file_type = dict(defaultextension=".pdf", filetypes=[("pdf file", "*.pdf")])
        file_path = filedialog.askopenfilename(**file_type)

        if file_path:
            self.path.append(file_path)
            file_name = file_path.split("/")[-1]
            self.text_box.delete(1.0, tk.END)
            self.text_box.insert(tk.INSERT, file_name)

    def clear_all(self):
        """
        Clear the query entry, text box, and reset the path.
        """
        self.query_entry.delete(0, tk.END)
        self.text_box.delete(1.0, tk.END)
        self.path.clear()

    def training(self):
        """
        Train the neural database client with the selected PDF file.
        """
        if not self.path:
            messagebox.showwarning("No File Selected", "Please select a PDF file before training.")
            return

        self.client.train(self.path[0])

        messagebox.showinfo("Training Complete", "Training is done!")

    def processing(self):
        """
        Process a user query and display the output in a new window.
        """
        question = self.query_entry.get()

        # When there is no query submitted by the user
        if not question:
            messagebox.showwarning("No Query", "Please enter a query.")
            return

        output = self.client.query(question)
        self.display_output(output)

    def display_output(self, output_data):
        """
        Display the output data in a new window.

        Args:
            output_data (str): The output text to be displayed.
        """
        output_window = tk.Toplevel(self.root)
        output_window.title("Output Data")
        output_window.geometry("500x500")

        output_text = tk.Text(output_window, wrap=tk.WORD, width=50, height=50)
        output_text.pack(padx=10, pady=10)
        output_text.insert(tk.END, output_data)


if __name__ == "__main__":
    """
    Initializing the main application window
    """

    # Calling the main application window
    win = tk.Tk()
    app = ThirdAIApp(win)
    win.mainloop()
