'''
The secrets module is used for generating cryptographically strong random numbers suitable for managing data such as passwords,
account authentication, security tokens, and related secrets.
'''
import string
import secrets

def secure_password_gen(passlength):
    password = ''.join((secrets.choice(string.ascii_letters) for i in range(passlength)))
    return password

n = int(input('Enter length of password : '))
print('Password generated is :', secure_password_gen(n))
