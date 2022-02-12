a = 0
while a<= 0 :
    number_to_divide = input("choose the number to divide -->")
    try :
        a = int(number_to_divide)
    except ValueError :
        a = 0
    if a <= 0 :
        print('choose a number grether than 0')
list_number_divided = []

for number in range(1,a + 1) :
    b = a % number
    if b == 0 :
        list_number_divided.append(number)
print('\nthe number ' + number_to_divide + ' can be divided by:')
for item in list_number_divided :
     print(f'{item}')
if len(list_number_divided) <= 2 :
    print(number_to_divide + ' is a prime number')