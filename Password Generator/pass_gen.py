import string as str
import secrets
class PasswordGenerator():

    @staticmethod
    def set_sequence(*conditions): #must have 4 conditions, one for each menber of the list possible_characters
        possible_characters=[str.ascii_lowercase, str.ascii_uppercase, str.digits, str.punctuation]
        sequence=""
        for x in range(len(conditions)):
            if conditions[x]:
                sequence+=possible_characters[x]
            else:
                pass
        return sequence


    @staticmethod
    def secure_password_gen(passlength=8, ):
        characters_for_password = PasswordGenerator.set_sequence(True, True, True, True)
        password = ''.join((secrets.choice(characters_for_password) for i in range(passlength)))
        return password

while True:
    print('Password generated is :', PasswordGenerator.secure_password_gen(int(input("Enter password lenght: "))))
