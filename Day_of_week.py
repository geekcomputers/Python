# Python program to Find day of 
# the week for a given date 
import calendar  #module of python to provide useful fucntions related to calendar
import datetime # module of python to get the date and time 


def findDay(date):
    born = datetime.datetime.strptime(date, '%d %m %Y').weekday() #this statement returns an integer corresponding to the day of the week
    return (calendar.day_name[born]) #this statement returns the corresponding day name to the integer generated in the previous statement


# Driver program 
date = '03 02 2019' #this is the input date
print(findDay(date))  # here we print the final output after calling the fucntion findday
