import string as str
import secrets
import random #this is the module used to generate random numbers on your given range
class PasswordGenerator():

    @staticmethod
    def gen_sequence(conditions): #must have  conditions (in a list format), for each member of the list possible_characters
        possible_characters=[str.ascii_lowercase, str.ascii_uppercase, str.digits, str.punctuation]
        sequence=""
        for x in range(len(conditions)):
            if conditions[x]:
                sequence+=possible_characters[x]
            else:
                pass
        return sequence

    @staticmethod
    def gen_password(sequence, passlength=8):
        password = ''.join((secrets.choice(sequence) for i in range(passlength)))
        return password

class Interface():
    has_characters={
    "lowercase":True,
    "uppercase":True,
    "digits":True,
    "punctuation":True
    }
    @classmethod
    def change_has_characters(cls, change):
        try:
            cls.has_characters[change] #to check if the specified key exists in the dicitonary
        except:
            print("Invalid")
        else:
            cls.has_characters[change]= not cls.has_characters[change] #automaticly changres to the oppesite value already there
            print(f"{change} is now set to {cls.has_characters[change]}")
    @classmethod
    def show_has_characters(cls):
        print(cls.has_characters) # print the output


    def generate_password(self, lenght):
        sequence = PasswordGenerator.gen_sequence(list(self.has_characters.values()))
        print(PasswordGenerator.gen_password(sequence, lenght))

def list_to_vertical_string(list):
    to_return =""
    for member in list:
        to_return += f"{member}\n"
    return to_return

class Run():
    def decide_operation(self):
        user_input = input(": ")
        try:
            int(user_input)
        except:
            Interface.change_has_characters(user_input)
        else:
            Interface().generate_password(int(user_input))
        finally:
            print("\n\n")



    def run(self):
        menu = \
f"""Welcome to the PassGen App!
Commands:
    generate password ->
    <lenght of the password>

commands to change the characters to be used to generate passwords:
{list_to_vertical_string(Interface.has_characters.keys())}
            """
        print(menu)
        while True:
            self.decide_operation()


Run().run()
