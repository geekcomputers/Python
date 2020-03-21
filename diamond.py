def draw_diamond(n):
    if n % 2 != 0:
        k = 1
        while k <= n:
            print(' '*int((n - k)/2)+'*'*k+' '*int((n - k)/2))
            k += 2

        j = 1
        while (n-2*j) >= 1:
            print(' ' *j + '*' * (n-2*j) + ' ' * (j))
            j += 1
    else:
        print('Not an odd number. Can\'t draw a diamond :(')


n = int(input('Enter an odd number: '))
draw_diamond(n)
