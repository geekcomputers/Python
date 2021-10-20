import re #regular expressions
import calendar  #module of python to provide useful fucntions related to calendar
import datetime # module of python to get the date and time 
import tkinter as tk
root = tk.Tk()
root.geometry("400x250+50+50")
user_input1=tk.StringVar()
def process_date(user_input):
    user_input=re.sub(r"/", " ", user_input) #substitute / with space
    user_input=re.sub(r"-", " ", user_input) #substitute - with space 
    return user_input

def find_day(date):
    born = datetime.datetime.strptime(date, '%d %m %Y').weekday() #this statement returns an integer corresponding to the day of the week
    return (calendar.day_name[born]) #this statement returns the corresponding day name to the integer generated in the previous statement

#To get the input from the user
#User may type 1/2/1999 or 1-2-1999
#To overcome those we have to process user input and make it standard to accept as defined by  calender and time module
def printt():
    user_input=user_input1.get()
    date=process_date(user_input)
    c="Day on " +user_input + "  is "+ find_day(date) 
    label2 = tk.Label(root,text=c,font=("Times new roman",20),fg='black').place(x=20,y=200)

lbl = tk.Label(root,text="Date --",font=("Ubuntu",20),fg="black").place(x=0,y=0.1,height=60,width=150)
lbl1 = tk.Label(root,text="(DD/MM/YYYY)",font=("Ubuntu",15),fg="Gray").place(x=120,y=0.1,height=60,width=150)
but = tk.Button(root,text="Check",command=printt,cursor="hand2",font=("Times new roman",40),fg="white",bg="black").place(x=50,y=130,height=50,width=300)
Date= tk.Entry(root,font=("Times new roman",20),textvariable=user_input1,bg="white",fg="black").place(x=30,y=50,height=40,width=340)    

root.mainloop()
