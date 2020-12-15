from tkinter import *
import qrcode
from PIL import Image, ImageTk
from resizeimage import resizeimage


class Qr:
    def __init__(self, root):
        pass
        self.root = root
        self.root.geometry("900x500+200+50")
        self.root.title("QR Generator | GS")
        self.root.resizable(False, False)

        title = Label(self.root, text="QR Code Generator", font=(
            "times new roman", 40), bg="#154360", fg="white").place(x=0, y=0, relwidth=1)

        # employee details window

        self.game_name = StringVar()
        self.game_ID = StringVar()
        self.game_password = StringVar()
        self.game_org = StringVar()

        emp_Frame = Frame(self.root, bd=2, relief=RIDGE, bg="white")
        emp_Frame.place(x=50, y=100, width=500, height=380)

        emp_title = Label(emp_Frame, text="Game Details", font=(
            "goudy old style", 30), bg="#154360", fg="white").place(x=0, y=0, relwidth=1)

        emp_label_details_name = Label(emp_Frame, text="Name", font=(
            "times new roman", 20, 'bold'), bg="white").place(x=10, y=60)
        emp_label_details_id = Label(emp_Frame, text="Room ID", font=(
            "times new roman", 20, 'bold'), bg="white").place(x=10, y=100)
        emp_label_details_password = Label(emp_Frame, text="Password", font=(
            "times new roman", 20, 'bold'), bg="white").place(x=10, y=140)
        emp_label_details_org = Label(emp_Frame, text="Organised By", font=(
            "times new roman", 20, 'bold'), bg="white").place(x=10, y=180)

        emp_entry_details_name = Entry(emp_Frame, textvariable=self.game_name, font=(
            "times new roman", 20, ), bg="lightyellow").place(x=200, y=60)
        emp_entry_details_id = Entry(emp_Frame, textvariable=self.game_ID, font=(
            "times new roman", 20, ), bg="lightyellow").place(x=200, y=100)
        emp_entry_details_password = Entry(emp_Frame, textvariable=self.game_password, font=(
            "times new roman", 20, ), bg="lightyellow").place(x=200, y=140)
        emp_entry_details_org = Entry(emp_Frame, textvariable=self.game_org, font=(
            "times new roman", 20, ), bg="lightyellow").place(x=200, y=180)

        btn_generate = Button(emp_Frame, text="QR Generate", command=self.generate, font=(
            "times new roman", 18, 'bold'), bg="#154360", fg="white").place(x=90, y=250, width=200, height=30)
        btn_clear = Button(emp_Frame, text="Clear", command=self.clear, font=("times new roman", 18, 'bold'),
                           bg="#154360", fg="white").place(x=300, y=250, width=150, height=30)

        self.msg = ""
        self.msg_label = Label(emp_Frame, text=self.msg, font=(
            "times new roman", 20, 'bold'), bg="white", fg="green")
        self.msg_label.place(x=0, y=310, relwidth=1)
        # qr scan image

        qr_Frame = Frame(self.root, bd=2, relief=RIDGE, bg="white")
        qr_Frame.place(x=600, y=100, width=250, height=380)

        qr_title = Label(qr_Frame, text="Game Code", font=(
            "goudy old style", 30), bg="#154360", fg="white").place(x=0, y=0, relwidth=1)

        self.qrcode = Label(qr_Frame, text="No QR \navailable", font=("times new roman", 18),
                            bg="#0C84D5", fg="white", bd=1, relief=RIDGE)
        self.qrcode.place(x=35, y=100, width=180, height=180)

    def generate(self):
        if(self.game_ID.get() == '' or self.game_name.get() == '' or self.game_org.get() == '' or self.game_password.get() == ''):
            self.msg = "All feilds are required!"
            self.msg_label.config(text=self.msg, fg="red")
        else:
            # updating notifications
            qr_data = (
                f"Name: {self.game_name.get()} \n Room ID: {self.game_ID.get()}\n Password: {self.game_password.get()}\n Organised By: {self.game_org.get()} \n")
            qr_code = qrcode.make(qr_data)
            qr_code=resizeimage.resize_cover(qr_code, [180,180])
            qr_code.save("qr/"+str(self.game_name)+ ".png")
            # image adding

            self.im = ImageTk.PhotoImage(qr_code)
            self.qrcode.config(image=self.im)
            self.msg = "QR Generated Successfully!"
            self.msg_label.config(text=self.msg, fg="green")

    def clear(self):
        self.game_name.set('')
        self.game_ID.set('')
        self.game_password.set('')
        self.game_org.set('')
        self.msg = ""
        self.msg_label.config(text=self.msg)
        self.qrcode.config(image='')


root = Tk()
obj = Qr(root)


root.mainloop()
