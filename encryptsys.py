import string
from random import randint


def decrypt():
    texto = input("Input the text to decrypt : ").split(".")
    abecedario = f"{string.printable}áéíóúÁÉÍÚÓàèìòùÀÈÌÒÙäëïöüÄËÏÖÜñÑ´"
    nummoves = int(texto[0])
    finalindexs = []
    textode1 = texto[1]
    abecedario2 = [abecedario[l] for l in range(len(abecedario))]
    textode2 = [textode1[letter] for letter in range(len(textode1))]
    indexs = [abecedario.index(textode1[index]) for index in range(len(textode1))]
    for _ in range(nummoves, 0):
        abecedario2 += abecedario2.pop(27)

    finalindexs.extend(value - nummoves for value in indexs)
    textofin = "".join(abecedario2[finalindex] for finalindex in finalindexs)

    print(textofin)


def encrypt():

    texto = input("Input the text to encrypt : ")
    abecedario = f"{string.printable}áéíóúÁÉÍÚÓàèìòùÀÈÌÒÙäëïöüÄËÏÖÜñÑ´"
    nummoves = randint(1, len(abecedario))
    abecedario2 = [abecedario[l] for l in range(len(abecedario))]
    texttoenc = [texto[let] for let in range(len(texto))]
    indexs = [abecedario2.index(letter) for letter in texto]
    for _ in range(nummoves):
        abecedario2 += abecedario2.pop(0)

    texto = []

    for index in indexs:
        texto.extend((abecedario2[index], "."))
    fintext = "".join(texto[letter2] for letter2 in range(0, len(texto), 2))

    fintext = f"{str(nummoves)}.{fintext}"

    print("\Encrypted text : " + fintext)


sel = input("What would you want to do?\n\n[1] Encrypt\n[2] Decrypt\n\n> ").lower()

if sel in ["1", "encrypt"]:
    encrypt()
elif sel in ["2", "decrypt"]:
    decrypt()
else:
    print("Unknown selection.")
