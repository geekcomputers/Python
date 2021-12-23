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
        """
        Displays the output of the calculation in a readonly text box.

        :param self: Instance of CalcFrame class.
        :type self: CalcFrame object.

        :param show:
        Text to be displayed in the output box, defaults to None. 
                     If no value is passed, it displays "0". 
                     This parameter is
        optional and can be omitted from call statement using *args syntax (see example).  

        :type show: str, int or float, optional

         .. note :: The default
        value for this parameter is set to None because if we want to display an empty string as output instead of 0 then we have two options - either pass an
        empty string ("") or leave this argument out completely (as shown below).  

                    Example 1 - passing "" as default argument value when
        function called without any arguments at all will result in displaying an empty string as output instead of 0;  ::     

                        def
        calc_output(self):                         # Function definition with no arguments passed during call statement execution...         # ...and hence
        default values for all parameters are used by function during its execution...        return ttk.Entry(self.cal_frame, textvariable
        """
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
        """
        Adds the number 7 to the output variable.

        If no value is in the output variable, it adds 7 to an empty string. Otherwise, it adds a new character (7)
        to 
        whatever is already in the output variable.

        Args: None

        Returns: None
        """
        current = self.out_var.get()
        if current == "":
            self.out_var.set(7)
        else:
            current += str(7)
            self.out_var.set(current)

    def press_8(self):
        """
        Prints the number 8 to the output field.
        If no number is present in the output field, it prints 8.
        Otherwise, it concatenates a string of numbers with
        8 and prints that value to the output field.
        """
        current = self.out_var.get()
        if current == "":
            self.out_var.set(8)
        else:
            current += str(8)
            self.out_var.set(current)

    def press_9(self):
        """
        This function is used to add 9 to the current value of the output variable.
        """
        current = self.out_var.get()
        if current == "":
            self.out_var.set(9)
        else:
            current += str(9)
            self.out_var.set(current)

    def press_4(self):
        """
        This function is used to add the number 4 to the current value in out_var.
        If there is no current value, it will set the output variable equal to 4.
        """
        current = self.out_var.get()
        if current == "":
            self.out_var.set(4)
        else:
            current += str(4)
            self.out_var.set(current)

    def press_5(self):
        """
        This function is used to add the number 5 to the current value in out_var.
        If there is no current value, it will set a new one with 5.
        Otherwise, it
        will concatenate the string of numbers with "5".
        """
        current = self.out_var.get()
        if current == "":
            self.out_var.set(5)
        else:
            current += str(5)
            self.out_var.set(current)

    def press_6(self):
        """
        This function is used to add the number 6 to the output box.
        """
        current = self.out_var.get()
        if current == "":
            self.out_var.set(6)
        else:
            current += str(6)
            self.out_var.set(current)

    def press_1(self):
        """
        This function takes a string and adds the number 1 to it. If no string is present, then it will set the output variable to 1.
        """
        current = self.out_var.get()
        if current == "":
            self.out_var.set(1)
        else:
            current += str(1)
            self.out_var.set(current)

    def press_2(self):
        """
        Adds a 2 to the output.

        If no number is present in the output, it adds a 2. Otherwise, it appends an additional 2 to the current value in the output.
        :param self: The object itself (always required)

        :returns: None
        """
        current = self.out_var.get()
        if current == "":
            self.out_var.set(2)
        else:
            current += str(2)
            self.out_var.set(current)

    def press_3(self):
        """
        This function is used to add the number 3 to the current value of self.out_var, which is a StringVar object that stores
        the output string. If there's
        no current value in self.out_var, then it will set its initial value as 3; otherwise,
        it will concatenate the new digit with the existing digits in
        self.out_var and update its current value accordingly.
        """
        current = self.out_var.get()
        if current == "":
            self.out_var.set(3)
        else:
            current += str(3)
            self.out_var.set(current)

    def press_0(self):
        """
        Adds a 0 to the output string.

        :param self: The object itself.

        :returns: None. 

            >>> press_0(self)

            Adds a 0 to the output string. 

            >>>
        press_0(self)()

            Returns None, as it is not meant to be called directly by the user.
        """
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
        """
        This function is used to add two numbers.
        """
        self.value1 = self.out_var.get()
        if self.value1 == "":
            messagebox.showwarning(
                "Operator Before Number", "Please Enter Number Before Operator"
            )
        else:
            self.out_var.set("")
            self.opr = "+"

    def press_min(self):
        """
        This function is used to subtract two numbers.
        It takes the first number as input and sets it to self.value1 variable, then clears the output screen
        and sets operator as '-'.
        """
        self.value1 = self.out_var.get()
        if self.value1 == "":
            messagebox.showwarning(
                "Operator Before Number", "Please Enter Number Before Operator"
            )
        else:
            self.out_var.set("")
            self.opr = "-"

    def press_mul(self):
        """
        This function is used to multiply the number.
        It takes two arguments, one is self and other one is none.
        If the value of first variable(self.value1)is
        empty then it will show warning message "Please Enter Number Before Operator" else it will set the operator as * and clear out_var(self.out_var).
        """
        self.value1 = self.out_var.get()
        if self.value1 == "":
            messagebox.showwarning(
                "Operator Before Number", "Please Enter Number Before Operator"
            )
        else:
            self.out_var.set("")
            self.opr = "*"

    def press_div(self):
        """
        This function is used to divide the numbers.
        It takes two inputs value1 and value2.
        If the first input is empty it shows a warning message box saying
        "Operator Before Number".
        Otherwise, it sets an operator as "/" and clears out_var.
        """
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
        """
        This function is used to perform calculation on two numbers.
        It takes the first number and second number as input from user and performs operation
        based on operator selected by user.
        """
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
        """
        This function creates a frame that contains the calculator buttons.

        :param self: The object itself.
        :type self: tkinter object.

        :returns: None.
        """
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
        """
        This function is used to set the geometry of the window.
        It takes 4 arguments:
            1. self - The current instance of the class.
            2. _com_width -
        The width of the computer screen in pixels (int).
            3. _com_height - The height of the computer screen in pixels (int). 
            4._my_width - The
        desired width for our window (int). 

        It returns nothing, but sets a string that contains all this information and sets it as our windows geometry
        using self's built-in method 'geometry'.

        Note: We use int() to convert these values from floats into integers because we cannot pass floats into
        tkinter's 'geometry' method, which accepts strings only!
        """
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
