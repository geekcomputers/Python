#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
# **************** Modules Require *****************#
from tkinter import *
from urllib.request import urlopen


# **************** Get IP commands *****************#
# control buttons
def get_wan_ip():
    try:
        # get ip from http://ipecho.net/plain as text
        wan_ip = urlopen('http://ipecho.net/plain').read().decode('utf-8')
        res.configure(text='Wan IP is : ' + wan_ip, fg='#600')
    except:
        res.configure(text='Problem in source : http://ipecho.net/plain', fg='red')


# get local ip
def get_local_ip():
    try:
        lan_ip = (socket.gethostbyname(socket.gethostname()))
        res.configure(text='Local IP is : ' + lan_ip, fg='#600')
    except:
        res.configure(text='Unkown Error', fg='#red')
    # **************** about control button *****************#


# show about info and change the button command and place
def about():
    global close_app, frame, info
    about_app.destroy()
    frame = Frame(root, width=350, height=2, bg='blue')
    frame.grid(row=2, column=0, columnspan=4)
    info = Label(root, text="""
    Practice Python 
    Take idea from here :
    https://github.com/geekcomputers/Python/blob/master/myip.py
    """, fg='#02F')
    info.grid(row=3, column=0, columnspan=4, padx=5)
    close_app = Button(root, text='Close', command=close_about, bg='#55F')
    close_app.grid(row=4, column=0, columnspan=4, pady=5)


# remove about info and remove close button then return about button in orignal place
def close_about():
    global frame, about_app, info
    info.destroy()
    frame.destroy()
    close_app.destroy()
    about_app = Button(root, text='about', command=about)
    about_app.grid(row=1, column=2, padx=5, pady=5, sticky=W)


# **************** Tkinter GUI *****************#
root = Tk()
root.title('Khaled programing practice')
# all buttons
res = Label(root, text='00.00.00.00', font=25)
res_wan_ip = Button(root, text='Get Wan IP', command=get_wan_ip)
res_local_ip = Button(root, text='Get Local IP', command=get_local_ip)
about_app = Button(root, text='about', command=about)
quit_app = Button(root, text='quit', command=quit, bg='#f40')
# method grid to install the button in window
res.grid(row=0, column=0, columnspan=4, sticky=N, padx=10, pady=5)
res_wan_ip.grid(row=1, column=0, padx=5, pady=5, sticky=W)
res_local_ip.grid(row=1, column=1, padx=5, pady=5, sticky=W)
about_app.grid(row=1, column=2, padx=5, pady=5, sticky=W)
quit_app.grid(row=1, column=3, padx=5, pady=5, sticky=E)
# run GUI/app
root.mainloop()
