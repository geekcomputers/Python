# ==================== Libraries ====================
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# ===================================================
# ==================== Classes ======================


class Inside:
    def __init__(self, parent):
        self.parent = parent
        # ----- Main Frame -----
        self.cal_frame = ttk.Frame(self.parent)
        self.cal_frame.grid(row=0, column=0)
        # ----------------------
        # ----- Variable For Main Output -----
        self.out_var = tk.StringVar()
        # ----- Operator Chooser -----
        self.opr = tk.StringVar()
        # ----- Values Holder -----
        self.value1 = tk.StringVar()
        self.value2 = tk.StringVar()
        # ------------------------------------
        self.output_box()  # <---------- Output Box Shower
        self.cal_buttons()  # <---------- Buttons On Calculator

    def output_box(self):
        show = ttk.Entry(
            self.cal_frame,
            textvariable=self.out_var,
            width=25,
            font=("calibri", 16),
            state="readonly",
        )
        show.grid(row=0, column=0, sticky=tk.W, ipady=6, ipadx=1, columnspan=4)
        show.focus()

    # ========== * Button Events * ========== < --- Sequence 789456123
    def press_7(self):
        current = self.out_var.get()
        if current == "":
            self.out_var.set(7)
        else:
            current += str(7)
            self.out_var.set(current)

    def press_8(self):
        current = self.out_var.get()
        if current == "":
            self.out_var.set(8)
        else:
            current += str(8)
            self.out_var.set(current)

    def press_9(self):
        current = self.out_var.get()
        if current == "":
            self.out_var.set(9)
        else:
            current += str(9)
            self.out_var.set(current)

    def press_4(self):
        current = self.out_var.get()
        if current == "":
            self.out_var.set(4)
        else:
            current += str(4)
            self.out_var.set(current)

    def press_5(self):
        current = self.out_var.get()
        if current == "":
            self.out_var.set(5)
        else:
            current += str(5)
            self.out_var.set(current)

    def press_6(self):
        current = self.out_var.get()
        if current == "":
            self.out_var.set(6)
        else:
            current += str(6)
            self.out_var.set(current)

    def press_1(self):
        current = self.out_var.get()
        if current == "":
            self.out_var.set(1)
        else:
            current += str(1)
            self.out_var.set(current)

    def press_2(self):
        current = self.out_var.get()
        if current == "":
            self.out_var.set(2)
        else:
            current += str(2)
            self.out_var.set(current)

    def press_3(self):
        current = self.out_var.get()
        if current == "":
            self.out_var.set(3)
        else:
            current += str(3)
            self.out_var.set(current)

    def press_0(self):
        current = self.out_var.get()
        if current == "":
            self.out_var.set(0)
        else:
            current += str(0)
            self.out_var.set(current)

    # ========== Operators Button Handling Function ==========
    def press_clear(self):
        self.out_var.set("")

    def press_reset(self):
        self.out_var.set("")

    def press_plus(self):
        self.value1 = self.out_var.get()
        if self.value1 == "":
            messagebox.showwarning(
                "Operator Before Number", "Please Enter Number Before Operator"
            )
        else:
            self.out_var.set("")
            self.opr = "+"

    def press_min(self):
        self.value1 = self.out_var.get()
        if self.value1 == "":
            messagebox.showwarning(
                "Operator Before Number", "Please Enter Number Before Operator"
            )
        else:
            self.out_var.set("")
            self.opr = "-"

    def press_mul(self):
        self.value1 = self.out_var.get()
        if self.value1 == "":
            messagebox.showwarning(
                "Operator Before Number", "Please Enter Number Before Operator"
            )
        else:
            self.out_var.set("")
            self.opr = "*"

    def press_div(self):
        self.value1 = self.out_var.get()
        if self.value1 == "":
            messagebox.showwarning(
                "Operator Before Number", "Please Enter Number Before Operator"
            )
        else:
            self.out_var.set("")
            self.opr = "/"

    # ==============================================
    # ========== ***** Equal Button Function ***** ==========
    def press_equal(self):
        self.value2 = self.out_var.get()
        if self.value2 == "":
            messagebox.showerror(
                "Second Number", "Please Enter Second Number To Perform Calculation"
            )
        else:

            try:
                if self.opr == "+":
                    self.value1 = int(self.value1)
                    self.value2 = int(self.value2)
                    result = self.value1 + self.value2
                    self.out_var.set(result)
                if self.opr == "-":
                    self.value1 = int(self.value1)
                    self.value2 = int(self.value2)
                    result = self.value1 - self.value2
                    self.out_var.set(result)
                if self.opr == "*":
                    self.value1 = int(self.value1)
                    self.value2 = int(self.value2)
                    result = self.value1 * self.value2
                    self.out_var.set(result)
                if self.opr == "/":
                    self.value1 = int(self.value1)
                    self.value2 = int(self.value2)
                    result = self.value1 / self.value2
                    self.out_var.set(result)

            except ValueError:
                messagebox.showinfo(
                    "Restart", "Please Close And Restart Application...Sorry"
                )

    def cal_buttons(self):
        # ===== Row 1 =====
        btn_c = tk.Button(
            self.cal_frame,
            text="Clear",
            width=6,
            height=2,
            bd=2,
            bg="#58a8e0",
            command=self.press_clear,
        )
        btn_c.grid(row=1, column=0, sticky=tk.W, pady=5)
        btn_div = tk.Button(
            self.cal_frame,
            text="/",
            width=6,
            height=2,
            bd=2,
            bg="#58a8e0",
            command=self.press_div,
        )
        btn_div.grid(row=1, column=1, sticky=tk.W)
        btn_mul = tk.Button(
            self.cal_frame,
            text="*",
            width=6,
            height=2,
            bd=2,
            bg="#58a8e0",
            command=self.press_mul,
        )
        btn_mul.grid(row=1, column=2, sticky=tk.E)
        btn_min = tk.Button(
            self.cal_frame,
            text="-",
            width=6,
            height=2,
            bd=2,
            bg="#58a8e0",
            command=self.press_min,
        )
        btn_min.grid(row=1, column=3, sticky=tk.E)
        # ===== Row 2 =====
        btn_7 = tk.Button(
            self.cal_frame,
            text="7",
            width=6,
            height=2,
            bd=2,
            bg="#90a9b8",
            command=self.press_7,
        )
        btn_7.grid(row=2, column=0, sticky=tk.W, pady=2)
        btn_8 = tk.Button(
            self.cal_frame,
            text="8",
            width=6,
            height=2,
            bd=2,
            bg="#90a9b8",
            command=self.press_8,
        )
        btn_8.grid(row=2, column=1, sticky=tk.W)
        btn_9 = tk.Button(
            self.cal_frame,
            text="9",
            width=6,
            height=2,
            bd=2,
            bg="#90a9b8",
            command=self.press_9,
        )
        btn_9.grid(row=2, column=2, sticky=tk.E)
        btn_plus = tk.Button(
            self.cal_frame,
            text="+",
            width=6,
            height=5,
            bd=2,
            bg="#58a8e0",
            command=self.press_plus,
        )
        btn_plus.grid(row=2, column=3, sticky=tk.E, rowspan=2)
        # ===== Row 3 =====
        btn_4 = tk.Button(
            self.cal_frame,
            text="4",
            width=6,
            height=2,
            bd=2,
            bg="#90a9b8",
            command=self.press_4,
        )
        btn_4.grid(row=3, column=0, sticky=tk.W, pady=2)
        btn_5 = tk.Button(
            self.cal_frame,
            text="5",
            width=6,
            height=2,
            bd=2,
            bg="#90a9b8",
            command=self.press_5,
        )
        btn_5.grid(row=3, column=1, sticky=tk.W)
        btn_6 = tk.Button(
            self.cal_frame,
            text="6",
            width=6,
            height=2,
            bd=2,
            bg="#90a9b8",
            command=self.press_6,
        )
        btn_6.grid(row=3, column=2, sticky=tk.E)
        # ===== Row 4 =====
        btn_1 = tk.Button(
            self.cal_frame,
            text="1",
            width=6,
            height=2,
            bd=2,
            bg="#90a9b8",
            command=self.press_1,
        )
        btn_1.grid(row=4, column=0, sticky=tk.W, pady=2)
        btn_2 = tk.Button(
            self.cal_frame,
            text="2",
            width=6,
            height=2,
            bd=2,
            bg="#90a9b8",
            command=self.press_2,
        )
        btn_2.grid(row=4, column=1, sticky=tk.W)
        btn_3 = tk.Button(
            self.cal_frame,
            text="3",
            width=6,
            height=2,
            bd=2,
            bg="#90a9b8",
            command=self.press_3,
        )
        btn_3.grid(row=4, column=2, sticky=tk.E)
        btn_equal = tk.Button(
            self.cal_frame,
            text="=",
            width=6,
            height=5,
            bd=2,
            bg="orange",
            command=self.press_equal,
        )
        btn_equal.grid(row=4, column=3, sticky=tk.E, rowspan=2)
        # ===== Row 5 =====
        btn_0 = tk.Button(
            self.cal_frame,
            text="0",
            width=14,
            height=2,
            bd=2,
            bg="#90a9b8",
            command=self.press_0,
        )
        btn_0.grid(row=5, column=0, sticky=tk.W, pady=2, columnspan=2)
        btn_reset = tk.Button(
            self.cal_frame,
            text="Reset",
            width=6,
            height=2,
            bd=2,
            bg="#90a9b8",
            command=self.press_reset,
        )
        btn_reset.grid(row=5, column=2, sticky=tk.E)


class Main(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ----- Title -----
        self.title("Calculator")
        # -----------------
        # ----- Geometry Settings -----
        self.geometry_settings()
        # -----------------------------

    def geometry_settings(self):
        _com_width = self.winfo_screenwidth()
        _com_height = self.winfo_screenheight()
        _my_width = 360
        _my_height = 350
        _x = int(_com_width / 2 - _my_width / 2)
        _y = int(_com_height / 2 - _my_height / 2)
        geo_string = (
            str(_my_width) + "x" + str(_my_height) + "+" + str(_x) + "+" + str(_y)
        )
        # ----- Setting Now -----
        self.geometry(geo_string)
        self.resizable(width=False, height=False)
        # -----------------------


# =================== Running The Application =======
if __name__ == "__main__":
    calculator = Main()
    Inside(calculator)
    calculator.mainloop()
