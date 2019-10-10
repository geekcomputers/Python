# Python program to Find day of 
# the week for a given date 
import calendar
import datetime


def findDay(date):
    born = datetime.datetime.strptime(date, '%d %m %Y').weekday()
    return (calendar.day_name[born])


# Driver program 
date = '03 02 2019'
print(findDay(date))
