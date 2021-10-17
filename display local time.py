import time
obj = time.localtime()
time = time.asctime(obj)
print(time)
hello=time.split()
day=hello[0]
day_=hello[2]
month=hello[1]
year=hello[4]
print("Date:",day,",",day_,"nd of",month,year)
print("time:", hello[3])
