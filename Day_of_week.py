# Python program to Find day of 
# the week for a given date 
import re #regular expressions
import calendar  #module of python to provide useful fucntions related to calendar
import datetime # module of python to get the date and time 

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
user_input=str(input("Enter date     "))
date=process_date(user_input)
print("Day on " +user_input + "  is "+ find_day(date) )


