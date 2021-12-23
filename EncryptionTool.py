# GGearing
# Simple encryption script for text
# This was one my first versions of this script
# 09/07/2017
from __future__ import print_function

import math

try:
    input = raw_input
except NameError:
    pass

key = int(math.pi * 1e14)
text = input("Enter text: ")
values = reverse = []


def encryptChar(target):
    # encrytion algorithm
    target = (((target + 42) * key) - 449)
    return target


def decryptChar(target):
    target = (((target + 449) / key) - 42)
    return target


def encrypt(input_text):
    """
    Encrypts a string of text by converting each character to its ASCII value and adding 1.
    """
    col_values = []
    for inp in input_text:
        current = ord(inp)
        current = encryptChar(current)
        col_values.append(current)
    return col_values


def decrypt(enc_text):
    """
    This function takes a list of encrypted characters and returns the decrypted message.
    """
    col_values = []
    for enc in enc_text:
        current = int(decryptChar(enc))
        current = chr(current)
        col_values.append(current)
    return col_values


def readAndDecrypt(filename):
    """
    This function takes a string of encrypted data and decrypts it using the given key.
    It returns the decrypted message as a list of characters.
    """
    file = open(filename, "r")
    data = file.read()
    datalistint = []
    actualdata = []
    datalist = data.split(" ")
    datalist.remove('')
    datalistint = [float(data) for data in datalist]
    for data in datalist:
        current1 = int(decryptChar(data))
        current1 = chr(current1)
        actualdata.append(current1)
    file.close()
    return actualdata


def readAndEncrypt(filename):
    """
    Reads a file and encrypts each character in the file using the function `encryptChar`.

    :param filename: The name of the file to be encrypted.
    :type
    filename: str.
    """
    file = open(filename, "r")
    data = file.read()
    datalist = list(data)
    encrypted_list = list()
    encrypted_list_str = list()
    for data in datalist:
        current = ord(data)
        current = encryptChar(current)
        encrypted_list.append(current)
    file.close()
    return encrypted_list


def readAndEncryptAndSave(inp_file, out_file):
    """
    Reads a file and encrypts its contents.

    :param inp_file: The name of the input file to be encrypted.
    :type inp_file: str

        :param out_file: The
    name of the output file to write the encrypted text.
        :type out_file: str

        Reads a given input file, encrypts its contents using AES encryption
    with 128 bit key and writes them into an output 
            text file named as specified by `out_file`. Returns None if successful else raises an
    exception on failure. 

            .. note :: This function is not tested yet! (TODO)
    """
    enc_list = readAndEncrypt(inp_file)
    output = open(out_file, "w")
    for enc in enc_list:
        output.write(str(enc) + " ")
    output.close()


def readAndDecryptAndSave(inp_file, out_file):
    """
    Reads a file and decrypts the contents.

    :param inp_file: The name of the input file to be decrypted.
    :type inp_file: str

        :param out_file: The
    name of the output file to be created with decrypted contents.
        :type out_file: str

        :returns list -- A list containing all of the lines from
    ``inp_file`` that have been decrypted using ``decrypt()`` function from this module, as strings.
    """
    dec_list = readAndDecrypt(inp_file)
    output = open(out_file, "w")
    for dec in dec_list:
        output.write(str(dec))
    output.close()


# encryption
for t in text:
    current = ord(t)
    current = encryptChar(current)
    values.append(current)

# decryption
for v in values:
    current = int(decryptChar(v))
    current = chr(current)
    reverse.append(current)
print(reverse)

# saves encrypted in txt file
output = open("encrypted.txt", "w")
for v in values:
    output.write(str(v) + " ")
output.close()

# read and decrypts
print(readAndDecrypt("encrypted.txt"))
