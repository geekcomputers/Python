import itertools

def generate_password_permutations(length):
    # Generate numeric password permutations of the given length
    digits = "0123456789"
    for combination in itertools.product(digits, repeat=length):
        password = "".join(combination)
        yield password

def password_cracker(target_password, max_length=8):
    # Try different password lengths and generate permutations
    for length in range(1, max_length + 1):
        password_generator = generate_password_permutations(length)
        for password in password_generator:
            if password == target_password:
                return password
    return None

if __name__ == "__main__":
    # Target numeric password (change this to the password you want to crack)
    target_password = "9133278"

    # Try cracking the password
    cracked_password = password_cracker(target_password)

    if cracked_password:
        print(f"Password successfully cracked! The password is: {cracked_password}")
    else:
        print("Password not found. Try increasing the max_length or target a different password.")

