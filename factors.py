import time
print('The factors of the number you type when prompted will be displayed')
time.sleep(4)
a = int(input('Type now //'))
b = a
while b > 0:
	if(a%b == 0):
		print(b)
	b -= 1
time.sleep(100)
