import calendar
import re
import sys
from datetime import date


def cleanDate(date_):
    date_.replace(' ', '')
    date_.replace(',', '')
    date_.replace('/', '')
    return date_


def getYearMonthDay(date_):
    day, year = map(int, re.findall(r'\d+', date_))
    month = ''.join(re.findall(r'[a-zA-Z]', date_)).capitalize()
    month_num = list(calendar.month_abbr).index(month[:3])

    return year, month_num, day


def numOfDays(year, month_num, day):
    days = date(year, month_num, day).timetuple()[-2]

    return days


date_ = sys.argv[1]

date_ = cleanDate(date_)
year, month_num, day = getYearMonthDay(date_)
print(numOfDays(year, month_num, day) - 1)
