import tkinter as tk
root = tk.Tk()
root.geometry("400x260+50+50")
root.title("Welcome to Letter Counter App")
message1 = tk.StringVar()
Letter1 = tk.StringVar()
def printt():
    """
    Prints the number of times a specified letter is in a message.
    :param message: The string to be searched for the letter.
    :type message: str
        :param
    letter: The string to be counted in the given string.
        :type letter: str
    """
    message=message1.get()
    letter=Letter1.get()
    message = message.lower()
    letter = letter.lower()

    # Get the count and display results.
    letter_count = message.count(letter)
    a = "your message has " + str(letter_count) + " " + letter + "'s in it."
    labl = tk.Label(root,text=a,font=('arial',15),fg='black').place(x=10,y=220)
lbl = tk.Label(root,text="Enter the Message--",font=('Ubuntu',15),fg='black').place(x=10,y=10)
lbl1 = tk.Label(root,text="Enter the Letter you want to count--",font=('Ubuntu',15),fg='black').place(x=10,y=80)
E1= tk.Entry(root,font=("arial",15),textvariable=message1,bg="white",fg="black").place(x=10,y=40,height=40,width=340)    
E2= tk.Entry(root,font=("arial",15),textvariable=Letter1,bg="white",fg="black").place(x=10,y=120,height=40,width=340)    
but = tk.Button(root,text="Check",command=printt,cursor="hand2",font=("Times new roman",30),fg="white",bg="black").place(x=10,y=170,height=40,width=380)
# print("In this app, I will count the number of times that a specific letter occurs in a message.")
# message = input("\nPlease enter a message: ")
# letter = input("Which letter would you like to count the occurrences of?: ")

root.mainloop()
