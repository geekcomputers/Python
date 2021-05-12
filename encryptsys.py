import string
from random import randint

def decrypt():
    texto = input("Input the text to decrypt : ").split(".")
    abecedario = string.printable + "áéíóúÁÉÍÚÓàèìòùÀÈÌÒÙäëïöüÄËÏÖÜñÑ´"
    abecedario2 = []
    nummoves = int(texto[0])
    indexs = []
    finalindexs = []
    textode1 = texto[1]
    textode2 = []

    for l in range(0, len(abecedario)):
        abecedario2.append(abecedario[l])

    for letter in range(0, len(textode1)):
        textode2.append(textode1[letter])

    for index in range(0, len(textode1)):
        indexs.append(abecedario.index(textode1[index]))

    for move in range(nummoves, 0):
        abecedario2 += abecedario2.pop(27)

    for value in indexs:
        newval = value - nummoves
        finalindexs.append(newval)

    textofin = ""

    for i in range(0, len(finalindexs)):
        textofin += abecedario2[finalindexs[i]]

    print(textofin)

def encrypt():

    texto = input("Input the text to encrypt : ")
    abecedario = string.printable + "áéíóúÁÉÍÚÓàèìòùÀÈÌÒÙäëïöüÄËÏÖÜñÑ´"
    abecedario2 = []
    nummoves = randint(1, len(abecedario))
    indexs = []

    texttoenc = []

    for l in range(0, len(abecedario)):
        abecedario2.append(abecedario[l])

    for let in range(0, len(texto)):
        texttoenc.append(texto[let])

    for letter in texto:
        indexs.append(abecedario2.index(letter))

    for move in range(0, nummoves):
        abecedario2 += abecedario2.pop(0)

    texto = []

    for i in range(0, len(indexs)):
        texto.append(abecedario2[indexs[i]])
        texto.append(".")

    fintext = ""

    for letter2 in range(0, len(texto), 2):
        fintext += texto[letter2]

    fintext = str(nummoves) + "." + fintext

    print("\Encrypted text : " + fintext)

sel = input("What would you want to do?\n\n[1] Encrypt\n[2] Decrypt\n\n> ").lower()

if sel in ["1", "encrypt"]:
    encrypt()
elif sel in ["2", "decrypt"]:
    decrypt()
else:
    print("Unknown selection.")
