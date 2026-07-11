#  ------------------------------------------------- ###
#  ------------------------------------------------- ###
#  ### Developed by TANMAY KHANDELWAL (aka Dude901). ###
#  _________________________________________________ ###
#  _________________________________________________ ###

import os
import webbrowser
from random import choices, randint
from tkinter import (
    END,
    HORIZONTAL,
    LEFT,
    Button,
    Entry,
    IntVar,
    Label,
    Scrollbar,
    StringVar,
    Text,
    Tk,
)

mycolor = "#262626"


class Case:
    def __init__(self, master: Tk) -> None:
        self.master = master
        self.test_case_counter = None  # kept for potential future use
        self.constraints = None
        # attributes created dynamically, but we keep them
        self.statement = None
        self.button1 = self.button2 = self.button3 = self.button4 = self.button5 = None
        self.button6 = self.button7 = self.button8 = self.button9 = self.button10 = None
        self.button_new = self.button_exit = self.copyright_label = None
        self.output = None
        self.y_scroll = self.x_scroll = None
        self.copy_button = self.generate_button = self.change_values_button = None
        self.done_button = self.button_exit_output = None
        self.test_case_count_label = self.test_case_count = None
        self.minimum_value_of_n = self.maximum_value_of_n = None
        self.min_max_values_of_n_label = None
        self.minimum_value_of_m = self.maximum_value_of_m = None
        self.min_max_values_of_m_label = None
        self.minimum_value_of_k = self.maximum_value_of_k = None
        self.min_max_values_of_k_label = None
        self.minimum_value_of_ai = self.maximum_value_of_ai = None
        self.min_max_values_of_ai_label = None
        self.minimum_value_of_bi = self.maximum_value_of_bi = None
        self.min_max_values_of_bi_label = None
        self.char_list_label = self.char_list = None
        self.back_btn = self.sub_btn = self.exit_btn = None
        # temporary variables for generation
        self.t = 0
        self.n_min = self.n_max = 0
        self.m_min = self.m_max = 0
        self.k_min = self.k_max = 0
        self.a_min = self.a_max = 0
        self.b_min = self.b_max = 0
        self.char_lis = []
        self.n = self.m = self.k = 0
        self.a = self.b = []

    def home(self) -> None:
        self.statement = Label(
            self.master,
            text="Select Test Case Type",
            fg="white",
            height=1,
            font=("calibre", 12, "normal"),
        )
        self.statement.configure(bg=mycolor)
        self.button1 = Button(
            self.master,
            justify=LEFT,
            text="T\nn   \nA1 A2 A3...An\nn   \nA1 A2 A3...An",
            width=13,
            fg="white",
            bd=3,
            command=lambda: Type1(self.master),
            bg="red",
            font="calibre",
        )
        self.button1.configure(background="grey20")
        self.button2 = Button(
            self.master,
            justify=LEFT,
            text="T\nn  m  \nA1 A2 A3...An\nn  m\nA1 A2 A3...An",
            fg="white",
            command=lambda: Type2(self.master),
            width=13,
            font="calibre",
            bd=3,
        )
        self.button2.configure(background="grey20")
        self.button3 = Button(
            self.master,
            justify=LEFT,
            text="T\nA1  B1\nA2  B2\n(t rows of)\n(A, B pair)",
            fg="white",
            command=lambda: Type3(self.master),
            width=13,
            font="calibre",
            bd=3,
        )
        self.button3.configure(background="grey20")
        self.button4 = Button(
            self.master,
            justify=LEFT,
            text="T\nn  m  \nA1 A2...An\nB1 B2...Bm\n...  ...",
            fg="white",
            command=lambda: Type4(self.master),
            width=13,
            font="calibre",
            bd=3,
        )
        self.button4.configure(background="grey20")
        self.button5 = Button(
            self.master,
            justify=LEFT,
            text="T\nn  m  k\nn  m  k\n(t rows of)\n(n m k  pair)",
            fg="white",
            command=lambda: Type5(self.master),
            width=13,
            font="calibre",
            bd=3,
        )
        self.button5.configure(background="grey20")
        self.button6 = Button(
            self.master,
            justify=LEFT,
            text="n * m (matrix)\nA1  A2...Am\nA1  A2...Am\n__   __ ... __\n"
            "A1  A2...Am",
            fg="white",
            command=lambda: Type6(self.master),
            width=13,
            font="calibre",
            bd=3,
        )
        self.button6.configure(background="grey20")
        self.button7 = Button(
            self.master,
            justify=LEFT,
            text="T\nn\nCustom string\n(ex: 0 1)\n(ex: + / -)",
            fg="white",
            command=lambda: Type7(self.master),
            width=13,
            font="calibre",
            bd=3,
        )
        self.button7.configure(background="grey20")
        self.button8 = Button(
            self.master,
            justify=LEFT,
            text="T\nn  m\nA1  B1\n...   ...\nAm  Bm",
            fg="white",
            command=lambda: Type8(self.master),
            width=13,
            font="calibre",
            bd=3,
        )
        self.button8.configure(background="grey20")
        self.button9 = Button(
            self.master,
            justify=LEFT,
            text='T\nCustom string\n(without "n")\n(ex: 0 1)\n(ex: + / -)',
            fg="white",
            command=lambda: Type9(self.master),
            width=13,
            font="calibre",
            bd=3,
        )
        self.button9.configure(background="grey20")
        self.button10 = Button(
            self.master,
            justify=LEFT,
            text="T\nn  k  m\nA1 A2...An\nn  k  m\nA1 A2...An",
            fg="white",
            command=lambda: Type10(self.master),
            width=13,
            font="calibre",
            bd=3,
        )
        self.button10.configure(background="grey20")
        self.button_new = Button(
            self.master,
            text=" ANOTHER TYPE ",
            fg="black",
            width=13,
            font="calibre",
            bd=3,
            command=self.newformat,
        )
        self.button_exit = Button(
            self.master,
            text=" EXIT ",
            fg="black",
            width=11,
            font="calibre",
            bd=3,
            command=self.master.destroy,
        )
        self.copyright_label = Button(
            self.master,
            text="© Dude901",
            fg="white",
            width=7,
            height=1,
            bd=3,
            command=lambda: webbrowser.open_new_tab("https://github.com/Tanmay-901"),
            font=("calibre", 6, "normal"),
        )
        self.copyright_label.configure(bg=mycolor)
        self.retrieve_home()

    def newformat(self) -> None:
        webbrowser.open_new_tab("https://forms.gle/UVdo6QMAwBNxa9Ln7")

    def forget_home(self) -> None:
        self.statement.place_forget()
        self.button1.grid_forget()
        self.button2.grid_forget()
        self.button3.grid_forget()
        self.button4.grid_forget()
        self.button5.grid_forget()
        self.button6.grid_forget()
        self.button7.grid_forget()
        self.button8.grid_forget()
        self.button9.grid_forget()
        self.button10.grid_forget()
        self.button_new.grid_forget()
        self.button_exit.grid_forget()

    def retrieve_home(self) -> None:
        self.statement.place(relx=0.39, rely=0.005)
        self.button1.grid(row=1, column=0, ipady=10, pady=27, padx=10)
        self.button2.grid(row=1, column=1, ipady=10, pady=27, padx=10)
        self.button3.grid(row=1, column=2, ipady=10, pady=27, padx=10)
        self.button4.grid(row=1, column=3, ipady=10, pady=27, padx=10)
        self.button5.grid(row=1, column=4, ipady=10, pady=27, padx=10)
        self.button6.grid(row=2, column=0, ipady=10, pady=13, padx=10)
        self.button7.grid(row=2, column=1, ipady=10, pady=13, padx=10)
        self.button8.grid(row=2, column=2, ipady=10, pady=13, padx=10)
        self.button9.grid(row=2, column=3, ipady=10, pady=13, padx=10)
        self.button10.grid(row=2, column=4, ipady=10, pady=13, padx=10)
        self.button_new.grid(row=3, column=1, ipady=10, pady=13, padx=10)
        self.button_exit.grid(row=3, column=3, ipady=10, pady=13, padx=10)
        self.copyright_label.place(relx=0.92, rely=0.005)

    def cpy(self) -> None:
        txt = self.output.get("1.0", END)
        self.master.clipboard_clear()
        self.master.clipboard_append(txt.strip())

    def done(self) -> None:
        self.try_forget()
        self.retrieve_home()

    def display(self) -> None:
        self.y_scroll = Scrollbar(self.master)
        self.x_scroll = Scrollbar(self.master, orient=HORIZONTAL)
        self.y_scroll.grid(row=0, column=11, sticky="NS", pady=(22, 0), padx=(0, 20))
        self.x_scroll.grid(
            row=1, sticky="EW", columnspan=10, padx=(20, 0), pady=(0, 30)
        )
        self.output = Text(
            self.master,
            height=12,
            bg="light cyan",
            width=82,
            yscrollcommand=self.y_scroll.set,
            xscrollcommand=self.x_scroll.set,
            wrap="none",
        )
        self.output.grid(
            row=0,
            column=0,
            columnspan=10,
            sticky="n",
            ipady=10,
            padx=(20, 0),
            pady=(22, 0),
        )
        self.y_scroll.config(command=self.output.yview)
        self.x_scroll.config(command=self.output.xview)
        self.copy_button = Button(
            self.master,
            text="COPY",
            fg="black",
            width=18,
            command=self.cpy,
            font="calibre",
            bd=3,
        )
        self.copy_button.grid(
            row=2, column=3, sticky="SW", ipady=10, pady=(10, 18), padx=15
        )
        self.generate_button = Button(
            self.master,
            text="RE-GENERATE",
            width=23,
            fg="black",
            command=self.generate,
            font="calibre",
            bd=3,
        )
        self.generate_button.grid(row=2, column=4, ipady=10, pady=(10, 18), padx=15)

        self.change_values_button = Button(
            self.master,
            text="CHANGE CONSTRAINT",
            fg="black",
            command=self.take_input,
            width=20,
            font="calibre",
            bd=3,
        )
        self.change_values_button.grid(row=2, column=5, ipady=10, pady=(10, 18), padx=5)
        self.done_button = Button(
            self.master,
            text="HOME",
            fg="black",
            command=self.done,
            width=20,
            font="calibre",
            bd=3,
        )
        self.done_button.grid(
            row=3, column=3, columnspan=2, ipady=10, pady=(10, 20), padx=5
        )
        self.button_exit_output = Button(
            self.master,
            text=" EXIT ",
            fg="black",
            width=20,
            font="calibre",
            bd=3,
            command=self.master.destroy,
        )
        self.button_exit_output.grid(
            row=3, column=4, columnspan=2, ipady=10, pady=(10, 20), padx=5
        )

    def try_forget(self) -> None:
        if self.output:
            self.output.grid_forget()
        if self.copy_button:
            self.copy_button.grid_forget()
        if self.generate_button:
            self.generate_button.grid_forget()
        if self.change_values_button:
            self.change_values_button.grid_forget()
        if self.done_button:
            self.done_button.grid_forget()
        if self.y_scroll:
            self.y_scroll.grid_forget()
        if self.x_scroll:
            self.x_scroll.grid_forget()
        if self.button_exit_output:
            self.button_exit_output.grid_forget()
        if self.constraints:
            self.constraints.grid_forget()

    def get_t(self, r: int) -> None:
        self.test_case_count_label = Label(
            self.master, text="T  = ", font=("calibre", 10, "bold"), width=17
        )
        self.test_case_count = Entry(
            self.master, textvariable=t, font=("calibre", 10, "normal")
        )
        self.test_case_count_label.grid(row=r, column=0, pady=20, ipady=1)
        self.test_case_count.grid(row=r, column=1)

    def get_n(self, r: int) -> None:
        self.minimum_value_of_n = Entry(
            self.master, textvariable=n_min, font=("calibre", 10, "normal")
        )
        self.min_max_values_of_n_label = Label(
            self.master, text=" <= n <=", font=("calibre", 10, "bold")
        )
        self.maximum_value_of_n = Entry(
            self.master, textvariable=n_max, font=("calibre", 10, "normal")
        )
        self.minimum_value_of_n.grid(row=r, column=0, padx=10, pady=10)
        self.min_max_values_of_n_label.grid(row=r, column=1, ipadx=5, ipady=1)
        self.maximum_value_of_n.grid(row=r, column=2, padx=(10, 10))

    def get_m(self, r: int) -> None:
        self.minimum_value_of_m = Entry(
            self.master, textvariable=m_min, font=("calibre", 10, "normal")
        )
        self.min_max_values_of_m_label = Label(
            self.master, text="<= m <=", font=("calibre", 10, "bold")
        )
        self.maximum_value_of_m = Entry(
            self.master, textvariable=m_max, font=("calibre", 10, "normal")
        )
        self.minimum_value_of_m.grid(row=r, column=0, padx=10, pady=10)
        self.min_max_values_of_m_label.grid(row=r, column=1, padx=10, ipadx=5, ipady=1)
        self.maximum_value_of_m.grid(row=r, column=2, padx=10)

    def get_k(self, r: int) -> None:
        self.minimum_value_of_k = Entry(
            self.master, textvariable=k_min, font=("calibre", 10, "normal")
        )
        self.min_max_values_of_k_label = Label(
            self.master, text=" <= k <=", font=("calibre", 10, "bold")
        )
        self.maximum_value_of_k = Entry(
            self.master, textvariable=k_max, font=("calibre", 10, "normal")
        )
        self.minimum_value_of_k.grid(row=r, column=0, pady=10)
        self.min_max_values_of_k_label.grid(row=r, column=1)
        self.maximum_value_of_k.grid(row=r, column=2)

    def get_a(self, r: int) -> None:
        self.minimum_value_of_ai = Entry(
            self.master, textvariable=a_min, font=("calibre", 10, "normal")
        )
        self.min_max_values_of_ai_label = Label(
            self.master, text=" <= Ai <=", font=("calibre", 10, "bold")
        )
        self.maximum_value_of_ai = Entry(
            self.master, textvariable=a_max, font=("calibre", 10, "normal")
        )
        self.minimum_value_of_ai.grid(row=r, column=0, padx=10, pady=10)
        self.min_max_values_of_ai_label.grid(row=r, column=1, ipadx=2, ipady=1)
        self.maximum_value_of_ai.grid(row=r, column=2)

    def get_b(self, r: int) -> None:
        self.minimum_value_of_bi = Entry(
            self.master, textvariable=b_min, font=("calibre", 10, "normal")
        )
        self.min_max_values_of_bi_label = Label(
            self.master, text=" <= Bi <= ", font=("calibre", 10, "bold")
        )
        self.maximum_value_of_bi = Entry(
            self.master, textvariable=b_max, font=("calibre", 10, "normal")
        )
        self.minimum_value_of_bi.grid(row=r, column=0, pady=10)
        self.min_max_values_of_bi_label.grid(row=r, column=1, padx=10)
        self.maximum_value_of_bi.grid(row=r, column=2, padx=10)

    def get_char_list(self, r: int) -> None:
        self.char_list_label = Label(
            self.master, text="  Characters :  ", font=("calibre", 10, "bold"), width=17
        )
        self.char_list = Entry(
            self.master, textvariable=char_lis, font=("calibre", 10, "normal"), width=43
        )
        self.char_list.insert(END, "(Space separated characters)")
        self.char_list.bind("<FocusIn>", lambda args: self.char_list.delete("0", "end"))
        self.char_list_label.grid(row=r, column=0, pady=10)
        self.char_list.grid(row=r, column=1, columnspan=2, padx=10)

    def show_button(self, r: int) -> None:
        self.back_btn = Button(
            self.master,
            text=" HOME ",
            command=lambda: self.forget_testcase_take_input_screen(1),
            font="calibre",
            bd=3,
        )
        self.sub_btn = Button(
            self.master, text=" GENERATE ", command=self.submit, font="calibre", bd=3
        )
        self.exit_btn = Button(
            self.master,
            text=" EXIT ",
            command=self.master.destroy,
            font="calibre",
            bd=3,
        )
        self.back_btn.grid(row=r, column=0, pady=(20, 20), ipady=1)
        self.sub_btn.grid(row=r, column=1, pady=(20, 20), ipady=1)
        self.exit_btn.grid(row=r, column=2, pady=(20, 20), ipady=1)
        self.copyright_label.place(relx=0.9, y=0)

    def submit(self) -> None:
        try:
            self.t = int(self.test_case_count.get())
            if self.t == 0 or self.t > 10000:
                return
        except ValueError, AttributeError:
            pass
        try:
            n1 = int(self.minimum_value_of_n.get())
            n2 = int(self.maximum_value_of_n.get())
            self.n_min = min(n1, n2)
            self.n_max = max(n1, n2)
            if self.n_min > self.n_max or self.n_max == 0 or self.n_max > 10000000:
                return
        except ValueError, AttributeError:
            pass
        try:
            m1 = int(self.minimum_value_of_m.get())
            m2 = int(self.maximum_value_of_m.get())
            self.m_min = min(m1, m2)
            self.m_max = max(m1, m2)
            if self.m_min > self.m_max or self.m_max == 0 or self.m_max > 10000000:
                return
        except ValueError, AttributeError:
            pass
        try:
            k1 = int(self.minimum_value_of_k.get())
            k2 = int(self.maximum_value_of_k.get())
            self.k_min = min(k1, k2)
            self.k_max = max(k1, k2)
            if self.k_min > self.k_max or self.k_max == 0 or self.k_max > 10000000:
                return
        except ValueError, AttributeError:
            pass
        try:
            a1 = int(self.minimum_value_of_ai.get())
            a2 = int(self.maximum_value_of_ai.get())
            self.a_min = min(a1, a2)
            self.a_max = max(a1, a2)
            if self.a_min > self.a_max or self.a_max == 0 or self.a_max > 10000000:
                return
        except ValueError, AttributeError:
            pass
        try:
            b1 = int(self.minimum_value_of_bi.get())
            b2 = int(self.maximum_value_of_bi.get())
            self.b_min = min(b1, b2)
            self.b_max = max(b1, b2)
            if self.b_min > self.b_max or self.b_max == 0 or self.b_max > 10000000:
                return
        except ValueError, AttributeError:
            pass
        try:
            self.char_lis = list(self.char_list.get().split())
            if not self.char_lis or self.char_lis[0] == "(Space":
                return
        except IndexError, ValueError, AttributeError:
            pass

        # additional sanity checks
        if (
            hasattr(self, "t")
            and hasattr(self, "n_max")
            and self.t * self.n_max > 10000000
        ):
            return
        if (
            hasattr(self, "m_max")
            and hasattr(self, "n_max")
            and self.m_max * self.n_max > 10000000
        ):
            return
        if (
            hasattr(self, "t")
            and hasattr(self, "m_max")
            and self.t * self.m_max > 10000000
        ):
            return

        self.forget_testcase_take_input_screen()
        self.display()
        self.generate()

    def forget_testcase_take_input_screen(self, check: int = 0) -> None:
        # try to forget all possible widgets
        for widget_name in (
            "test_case_count_label",
            "test_case_count",
            "minimum_value_of_n",
            "min_max_values_of_n_label",
            "maximum_value_of_n",
            "minimum_value_of_ai",
            "min_max_values_of_ai_label",
            "maximum_value_of_ai",
            "minimum_value_of_bi",
            "min_max_values_of_bi_label",
            "maximum_value_of_bi",
            "minimum_value_of_m",
            "min_max_values_of_m_label",
            "maximum_value_of_m",
            "minimum_value_of_k",
            "min_max_values_of_k_label",
            "maximum_value_of_k",
            "char_list_label",
            "char_list",
            "constraints",
        ):
            widget = getattr(self, widget_name, None)
            if widget:
                try:
                    widget.grid_forget()
                except:
                    pass
        # clear char_list if exists
        if self.char_list:
            try:
                self.char_list.delete("0", "end")
            except:
                pass
        # forget buttons
        for btn in ("sub_btn", "back_btn", "exit_btn"):
            widget = getattr(self, btn, None)
            if widget:
                try:
                    widget.grid_forget()
                except:
                    pass

        if check:
            self.retrieve_home()

    # Placeholder for generate method – will be overridden in subclasses
    def generate(self) -> None:
        pass

    # Placeholder for take_input – will be overridden
    def take_input(self) -> None:
        pass


# ---------- Type1 to Type10 ----------
class Type1(Case):
    def __init__(self, master: Tk) -> None:
        super().__init__(master)
        self.forget_home()
        self.take_input()

    def take_input(self) -> None:
        self.try_forget()
        self.get_t(0)
        self.get_n(1)
        self.get_a(2)
        self.show_button(3)

    def generate(self) -> None:
        self.forget_testcase_take_input_screen()
        self.output.delete("1.0", END)
        self.output.insert(END, self.t)
        self.output.insert(END, "\n")
        for _ in range(self.t):
            n = randint(self.n_min, self.n_max)
            self.output.insert(END, n)
            self.output.insert(END, "\n")
            arr = [randint(self.a_min, self.a_max) for _ in range(n)]
            self.output.insert(END, arr)
            self.output.insert(END, "\n")


class Type2(Case):
    def __init__(self, master: Tk) -> None:
        super().__init__(master)
        self.forget_home()
        self.take_input()

    def take_input(self) -> None:
        self.try_forget()
        self.get_t(0)
        self.get_n(1)
        self.get_m(2)
        self.get_a(3)
        self.show_button(4)

    def generate(self) -> None:
        self.forget_testcase_take_input_screen()
        self.output.delete("1.0", END)
        self.output.insert(END, self.t)
        self.output.insert(END, "\n")
        for _ in range(self.t):
            n = randint(self.n_min, self.n_max)
            m = randint(self.m_min, self.m_max)
            self.output.insert(END, n)
            self.output.insert(END, " ")
            self.output.insert(END, m)
            self.output.insert(END, "\n")
            arr = [randint(self.a_min, self.a_max) for _ in range(n)]
            self.output.insert(END, arr)
            self.output.insert(END, "\n")


class Type3(Case):
    def __init__(self, master: Tk) -> None:
        super().__init__(master)
        self.forget_home()
        self.take_input()

    def take_input(self) -> None:
        self.try_forget()
        self.get_t(0)
        self.get_a(1)
        self.get_b(2)
        self.show_button(3)

    def generate(self) -> None:
        self.forget_testcase_take_input_screen()
        self.output.delete("1.0", END)
        self.output.insert(END, self.t)
        self.output.insert(END, "\n")
        for _ in range(self.t):
            a = randint(self.a_min, self.a_max)
            b = randint(self.b_min, self.b_max)
            self.output.insert(END, a)
            self.output.insert(END, " ")
            self.output.insert(END, b)
            self.output.insert(END, "\n")


class Type4(Case):
    def __init__(self, master: Tk) -> None:
        super().__init__(master)
        self.forget_home()
        self.take_input()

    def take_input(self) -> None:
        self.try_forget()
        self.get_t(0)
        self.get_n(1)
        self.get_m(2)
        self.get_a(3)
        self.get_b(4)
        self.show_button(5)

    def generate(self) -> None:
        self.forget_testcase_take_input_screen()
        self.output.delete("1.0", END)
        self.output.insert(END, self.t)
        self.output.insert(END, "\n")
        for _ in range(self.t):
            n = randint(self.n_min, self.n_max)
            m = randint(self.m_min, self.m_max)
            self.output.insert(END, n)
            self.output.insert(END, " ")
            self.output.insert(END, m)
            self.output.insert(END, "\n")
            arr_a = [randint(self.a_min, self.a_max) for _ in range(n)]
            arr_b = [randint(self.b_min, self.b_max) for _ in range(m)]
            self.output.insert(END, arr_a)
            self.output.insert(END, "\n")
            self.output.insert(END, arr_b)
            self.output.insert(END, "\n")


class Type5(Case):
    def __init__(self, master: Tk) -> None:
        super().__init__(master)
        self.forget_home()
        self.take_input()

    def take_input(self) -> None:
        self.try_forget()
        self.get_t(0)
        self.get_n(1)
        self.get_m(2)
        self.get_k(3)
        self.show_button(4)

    def generate(self) -> None:
        self.forget_testcase_take_input_screen()
        self.output.delete("1.0", END)
        self.output.insert(END, self.t)
        self.output.insert(END, "\n")
        for _ in range(self.t):
            n = randint(self.n_min, self.n_max)
            m = randint(self.m_min, self.m_max)
            k = randint(self.k_min, self.k_max)
            self.output.insert(END, n)
            self.output.insert(END, " ")
            self.output.insert(END, m)
            self.output.insert(END, " ")
            self.output.insert(END, k)
            self.output.insert(END, "\n")


class Type6(Case):
    def __init__(self, master: Tk) -> None:
        super().__init__(master)
        self.forget_home()
        self.take_input()

    def take_input(self) -> None:
        self.try_forget()
        self.constraints = Label(
            self.master,
            text="Enter Constraints",
            fg="white",
            height=1,
            font=("calibre", 12, "normal"),
        )
        self.constraints.configure(bg=mycolor)
        self.constraints.grid(row=0, column=1)
        self.get_n(1)
        self.get_m(2)
        self.get_a(3)
        self.show_button(4)

    def generate(self) -> None:
        self.forget_testcase_take_input_screen()
        self.output.delete("1.0", END)
        n = randint(self.n_min, self.n_max)
        m = randint(self.m_min, self.m_max)
        self.output.insert(END, n)
        self.output.insert(END, " ")
        self.output.insert(END, m)
        self.output.insert(END, "\n")
        for _ in range(n):
            row = [randint(self.a_min, self.a_max) for _ in range(m)]
            self.output.insert(END, row)
            self.output.insert(END, "\n")


class Type7(Case):
    def __init__(self, master: Tk) -> None:
        super().__init__(master)
        self.forget_home()
        self.take_input()

    def take_input(self) -> None:
        self.try_forget()
        self.get_t(0)
        self.get_char_list(1)
        self.get_n(2)
        self.show_button(3)

    def generate(self) -> None:
        self.forget_testcase_take_input_screen()
        self.output.delete("1.0", END)
        self.output.insert(END, self.t)
        self.output.insert(END, "\n")
        for _ in range(self.t):
            n = randint(self.n_min, self.n_max)
            self.output.insert(END, n)
            self.output.insert(END, "\n")
            s = "".join(choices(self.char_lis, k=n))
            self.output.insert(END, s)
            self.output.insert(END, "\n")


class Type8(Case):
    def __init__(self, master: Tk) -> None:
        super().__init__(master)
        self.forget_home()
        self.take_input()

    def take_input(self) -> None:
        self.try_forget()
        self.get_t(0)
        self.get_n(1)
        self.get_m(2)
        self.get_a(3)
        self.get_b(4)
        self.show_button(5)

    def generate(self) -> None:
        self.forget_testcase_take_input_screen()
        self.output.delete("1.0", END)
        self.output.insert(END, self.t)
        self.output.insert(END, "\n")
        for _ in range(self.t):
            n = randint(self.n_min, self.n_max)
            m = randint(self.m_min, self.m_max)
            self.output.insert(END, n)
            self.output.insert(END, " ")
            self.output.insert(END, m)
            self.output.insert(END, "\n")
            for _ in range(m):
                a = randint(self.a_min, self.a_max)
                b = randint(self.b_min, self.b_max)
                self.output.insert(END, a)
                self.output.insert(END, " ")
                self.output.insert(END, b)
                self.output.insert(END, "\n")


class Type9(Case):
    def __init__(self, master: Tk) -> None:
        super().__init__(master)
        self.forget_home()
        self.take_input()

    def take_input(self) -> None:
        self.try_forget()
        self.get_t(0)
        self.get_char_list(1)
        self.get_n(2)
        self.show_button(3)

    def generate(self) -> None:
        self.forget_testcase_take_input_screen()
        self.output.delete("1.0", END)
        self.output.insert(END, self.t)
        self.output.insert(END, "\n")
        for _ in range(self.t):
            n = randint(self.n_min, self.n_max)
            s = "".join(choices(self.char_lis, k=n))
            self.output.insert(END, s)
            self.output.insert(END, "\n")


class Type10(Case):
    def __init__(self, master: Tk) -> None:
        super().__init__(master)
        self.forget_home()
        self.take_input()

    def take_input(self) -> None:
        self.try_forget()
        self.get_t(0)
        self.get_n(1)
        self.get_k(2)
        self.get_m(3)
        self.get_a(4)
        self.show_button(5)

    def generate(self) -> None:
        self.forget_testcase_take_input_screen()
        self.output.delete("1.0", END)
        self.output.insert(END, self.t)
        self.output.insert(END, "\n")
        for _ in range(self.t):
            n = randint(self.n_min, self.n_max)
            k = randint(self.k_min, self.k_max)
            m = randint(self.m_min, self.m_max)
            self.output.insert(END, n)
            self.output.insert(END, " ")
            self.output.insert(END, k)
            self.output.insert(END, " ")
            self.output.insert(END, m)
            self.output.insert(END, "\n")
            arr = [randint(self.a_min, self.a_max) for _ in range(n)]
            self.output.insert(END, arr)
            self.output.insert(END, "\n")


if __name__ == "__main__":
    root = Tk()
    root.title("TEST CASE GENERATOR")
    root.configure(bg=mycolor)

    if os.environ.get("DISPLAY", "") == "":
        print("no display found, using:0,0")
        os.environ.__setitem__("DISPLAY", ":0.0")
    else:
        print("found display")

    t = IntVar()
    n_min = IntVar()
    n_max = IntVar()
    m_min = IntVar()
    m_max = IntVar()
    k_min = IntVar()
    k_max = IntVar()
    a_min = IntVar()
    a_max = IntVar()
    b_min = IntVar()
    b_max = IntVar()
    char_lis = StringVar()

    case = Case(root)
    case.home()

    root.mainloop()
