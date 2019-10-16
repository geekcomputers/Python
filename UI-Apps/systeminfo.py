import platform
import tkinter
import sys
#------------------main code-----------------------
#initializing the main UI object
top = tkinter.Tk()
#setting title of the App
top.title('System Info')
#restricting the resizable property
top.resizable(0,0)

F1 = tkinter.Frame(top)
F2 = tkinter.Frame(top)
F3 = tkinter.Frame(top)
F4 = tkinter.Frame(top)
F5 = tkinter.Frame(top)
F6 = tkinter.Frame(top)
F7 = tkinter.Frame(top)
F8 = tkinter.Frame(top)
f = "Verdana 15"
L1 = tkinter.Label(F1, text="System Info", fg="black", font="Verdana 15 bold")
#------------------Architecture-------------------------
L2 = tkinter.Label(F2, text="Architecture", fg="black", font=f)
L3 = tkinter.Label(F2, text = platform.architecture(), fg="black", font=f)
#------------------Machine-------------------------
L4 = tkinter.Label(F3, text="Machine", fg="black", font=f)
L5 = tkinter.Label(F3, text = platform.machine(), fg="black", font=f) 
#------------------Network-------------------------
L6 = tkinter.Label(F4, text="Computer Name", fg="black", font=f)
L7 = tkinter.Label(F4, text = platform.node(), fg="black", font=f) 
#------------------Platform-------------------------
L8 = tkinter.Label(F5, text="OS", fg="black", font=f)
L9 = tkinter.Label(F5, text = platform.platform(), fg="black", font=f)
#------------------Processor-------------------------
L10 = tkinter.Label(F6, text="Processor", fg="black", font=f)
L11 = tkinter.Label(F6, text = platform.processor(), fg="black", font=f)
#------------------Sysytem-------------------------
L12 = tkinter.Label(F7, text="System", fg="black", font=f)
L13 = tkinter.Label(F7, text = platform.system(), fg="black", font=f)



F1.pack(anchor = 'center', pady=20)
F2.pack(anchor = 'center', pady=5)
F6.pack(anchor = 'center', pady=5)
F3.pack(anchor = 'center', pady=5)
F4.pack(anchor = 'center', pady=5)
F7.pack(anchor = 'center', pady=5)
F5.pack(anchor = 'center', pady=5)
F8.pack(anchor = 'center', pady=5)
L1.pack(side="left")
L2.pack(side="left",padx=15)
L3.pack(side="left",padx=15)
L4.pack(side="left",padx=15)
L5.pack(side="left",padx=15)
L6.pack(side="left",padx=15)
L7.pack(side="left",padx=15)
L8.pack(side="left",padx=15)
L9.pack(side="left",padx=15)
L10.pack(side="left",padx=15)
L11.pack(side="left",padx=15)
L12.pack(side="left",padx=15)
L13.pack(side="left",padx=15)
top.mainloop()