# ==================== Importing Libraries ====================
# =============================================================
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showerror
from tkinter.scrolledtext import ScrolledText

# =============================================================


class Main(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Alphacrypter")
        # ----- Setting Geometry -----
        self.geometry_settings()

    def geometry_settings(self):
        _com_scr_w = self.winfo_screenwidth()
        _com_scr_h = self.winfo_screenheight()
        _my_w = 300
        _my_h = 450
        # ----- Now Getting X and Y Coordinates
        _x = int(_com_scr_w / 2 - _my_w / 2)
        _y = int(_com_scr_h / 2 - _my_h / 2)
        _geo_string = str(_my_w) + "x" + str(_my_h) + "+" + str(_x) + "+" + str(_y)
        self.geometry(_geo_string)
        # ----- Geometry Setting Completed Now Disabling Resize Screen Button -----
        self.resizable(width=False, height=False)


class Notebook:
    def __init__(self, parent):
        self.parent = parent
        # ========== Data Key ==========
        self.data_dic = {
            "a": "q",
            "b": "w",
            "c": "e",
            "d": "r",
            "e": "t",
            "f": "y",
            "g": "u",
            "h": "i",
            "i": "o",
            "j": "p",
            "k": "a",
            "l": "s",
            "m": "d",
            "n": "f",
            "o": "g",
            "p": "h",
            "q": "j",
            "r": "k",
            "s": "l",
            "t": "z",
            "u": "x",
            "v": "c",
            "w": "v",
            "x": "b",
            "y": "n",
            "z": "m",
            "1": "_",
            "2": "-",
            "3": "|",
            "4": "?",
            "5": "*",
            "6": "!",
            "7": "@",
            "8": "#",
            "9": "$",
            "0": "~",
            ".": "/",
            ",": "+",
            " ": "&",
        }
        # ==============================
        # ----- Notebook With Two Pages -----
        self.nb = ttk.Notebook(self.parent)
        self.page1 = ttk.Frame(self.nb)
        self.page2 = ttk.Frame(self.nb)
        self.nb.add(self.page1, text="Encrypt The Words")
        self.nb.add(self.page2, text="Decrypt The Words")
        self.nb.pack(expand=True, fill="both")
        # ----- LabelFrames -----
        self.page1_main_label = ttk.LabelFrame(
            self.page1, text="Encrypt Any Text"
        )  # <----- Page1 LabelFrame1
        self.page1_main_label.grid(row=0, column=0, pady=20, padx=2, ipadx=20)
        self.page1_output_label = ttk.LabelFrame(self.page1, text="Decrypted Text")
        self.page1_output_label.grid(row=1, column=0, pady=10, padx=2)

        self.page2_main_label = ttk.LabelFrame(
            self.page2, text="Decrypt Any Text"
        )  # <----- Page1 LabelFrame1
        self.page2_main_label.grid(row=0, column=0, pady=20, padx=2, ipadx=20)
        self.page2_output_label = ttk.LabelFrame(self.page2, text="Real Text")
        self.page2_output_label.grid(row=1, column=0, pady=10, padx=2)
        # <---Scrolled Text Global
        self.decrypted_text_box = ScrolledText(
            self.page1_output_label, width=30, height=5, state="normal"
        )
        self.decrypted_text_box.grid(row=1, column=0, padx=2, pady=10)

        self.text_box = ScrolledText(
            self.page2_output_label, width=30, height=5, state="normal"
        )
        self.text_box.grid(row=1, column=0, padx=2, pady=10)
        # ----- Variables -----
        self.user_text = tk.StringVar()
        self.decrypted_user_text = tk.StringVar()

        self.user_text2 = tk.StringVar()
        self.real_text = tk.StringVar()
        # ----- Getting Inside Page1 -----
        self.page1_inside()
        self.page2_inside()

    def page1_inside(self):
        style = ttk.Style()
        user_text_label = ttk.Label(
            self.page1_main_label, text="Enter Your Text Here : ", font=("", 14)
        )
        user_text_label.grid(row=0, column=0, pady=10)
        user_entry_box = ttk.Entry(
            self.page1_main_label, width=35, textvariable=self.user_text
        )
        user_entry_box.grid(row=1, column=0)
        style.configure(
            "TButton",
            foreground="black",
            background="white",
            relief="groove",
            font=("", 12),
        )
        encrypt_btn = ttk.Button(
            self.page1_main_label,
            text="Encrypt Text",
            style="TButton",
            command=self.encrypt_now,
        )
        encrypt_btn.grid(row=2, column=0, pady=15)

    # ---------- Page1 Button Binding Function ----------

    def encrypt_now(self):
        user_text = self.user_text.get()
        if user_text == "":
            showerror(
                "Nothing Found", "Please Enter Something In Entry Box To Encrypt...!"
            )
            return
        else:
            self.decrypted_user_text = self.backend_work("Encrypt", user_text)
            self.decrypted_text_box.insert(tk.INSERT, self.decrypted_user_text, tk.END)

    # --------------------------------------------------Binding Functions of Page1 End Here
    # Page2 ------------------>
    def page2_inside(self):
        style = ttk.Style()
        user_text_label = ttk.Label(
            self.page2_main_label, text="Enter Decrypted Text Here : ", font=("", 14)
        )
        user_text_label.grid(row=0, column=0, pady=10)
        user_entry_box = ttk.Entry(
            self.page2_main_label, width=35, textvariable=self.user_text2
        )
        user_entry_box.grid(row=1, column=0)
        style.configure(
            "TButton",
            foreground="black",
            background="white",
            relief="groove",
            font=("", 12),
        )
        encrypt_btn = ttk.Button(
            self.page2_main_label,
            text="Decrypt Text",
            style="TButton",
            command=self.decrypt_now,
        )
        encrypt_btn.grid(row=2, column=0, pady=15)
        # ---------- Page1 Button Binding Function ----------

    def decrypt_now(self):
        user_text = self.user_text2.get()
        if user_text == "":
            showerror(
                "Nothing Found", "Please Enter Something In Entry Box To Encrypt...!"
            )
            return
        else:
            self.real_text = self.backend_work("Decrypt", user_text)
            self.text_box.insert(tk.INSERT, self.real_text, tk.END)

    def backend_work(self, todo, text_coming):
        text_to_return = ""
        if todo == "Encrypt":
            try:
                text_coming = str(
                    text_coming
                ).lower()  # <----- Lowering the letters as dic in lower letter
                for word in text_coming:
                    for key, value in self.data_dic.items():
                        if word == key:
                            # print(word, " : ", key)
                            text_to_return += value

            except ValueError:
                showerror("Unknown", "Something Went Wrong, Please Restart Application")

            return text_to_return
        elif todo == "Decrypt":
            try:
                text_coming = str(text_coming).lower()
                for word in text_coming:
                    for key, value in self.data_dic.items():
                        if word == value:
                            text_to_return += key

            except ValueError:
                showerror("Unknown", "Something Went Wrong, Please Restart Application")

            return text_to_return

        else:
            showerror("No Function", "Function Could not get what to do...!")


# =============================================================
# ==================== Classes End Here ... ! =================


if __name__ == "__main__":
    run = Main()
    Notebook(run)
    run.mainloop()
