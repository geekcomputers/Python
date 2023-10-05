kilo = float (input("kilonuzu giriniz(örnek: 84.9): "))
boy = float (input("Boyunuzu m cinsinden giriniz: "))

vki = (kilo / (boy**2))

if vki < 18.5:
    print(f"vucut kitle indeksiniz:  {vki} zayıfsınız.")
elif vki < 25:
    print (f"vucut kitle indeksiniz: {vki} normalsiniz.")
elif vki < 30:
    print (f"vucut kitle indeksiniz: {vki} fazla kilolusunuz.")
elif vki < 35:
    print (f"vucut kitle indeksiniz: {vki} 1. derece obezsiniz")
elif vki < 40:
    print (f"vucut kitle indeksiniz: {vki} 2.derece obezsiniz.")
elif vki >40:
    print (f"vucut kitle indeksiniz: {vki} 3.derece obezsiniz.")
else:
    print("Yanlış değer girdiniz.")
