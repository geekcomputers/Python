# A program to write a number in words
# Eg:
# 61893: Sixty One Thousand Eight Hundred Ninety Three

__import__('os').system('cls')


Y = "\033[38;2;255;200;0m"
W = "\033[38;2;212;212;212;0m"
B = "\033[38;2;108;180;238m;0m"

groupedList = []
name = ""
nameList = []

numDict = {
        1 : "One", 2 : "Two", 3 : "Three", 4 : "Four", 5 : "Five",
        6 : "Six", 7 : "Seven", 8 : "Eight", 9 : "Nine", 10 : "Ten",
        11 : "Eleven", 12 : "Twelve", 13 : "Thirteen", 14 : "Fourteen", 15 : "Fifteen",
        16 : "Sixteen", 17 : "Seventeen", 18 : "Eighteen", 19 : "Ninteen", 20 : "Twenty",
        30 : "Thirty", 40 : "Forty", 50 : "Fifty", 60 : "Sixty", 70 : "Seventy", 
        80 : "Eighty", 90 : "Ninety"
}

digits = {
        "1" : "One", "2" : "Two", "3" : "Three", "4" : "Four", "5" : "Five",
        "6" : "Six", "7" : "Seven", "8" : "Eight", "9" : "Nine", "0" : "Zero"
}

placeValueDict = {
        1 : "",
        2 : "Thousand",
        3 : "Million",
        4 : "Billion",
        5 : "Trillion",
        6 : "Quadrillion",
        7 : "Quintillion",
        8 : "Sextillion",
        9 : "Septilion",
        10 : "Octillion"
}

print("Maximum Input: 999,999,999,999,999,999,999,999,999,999")
print("Minimum Input: -999,999,999,999,999,999,999,999,999,999\n")

isNegative = False

while True:
        num = input(f"Enter a number: {Y}")
        print(f"{W}", end="")

        try:
                splittedNum = num.split(".")

                splittedNum[0] = splittedNum[0].replace(" ", "")
                if len(splittedNum) == 2:
                        splittedNum[1] = splittedNum[1].replace(" ", "")
                        splittedNum[1] = splittedNum[1].rstrip("0")

                        if splittedNum[1] == "":
                                splittedNum.remove("")

                num = int(splittedNum[0])

                if len(splittedNum) == 1:
                        placeholder = splittedNum[0]
                        placeholder = int(placeholder)
                else:
                        placeholder = splittedNum[0] + "." + splittedNum[1]
                        placeholder = float(placeholder)

                if num >= 1000000000000000000000000000000 or num <= -1000000000000000000000000000000:
                        print("Input out of range\n")
                else:
                        if num < 0:
                                isNegative = True
                                num = num * (-1)
                        break
        except ValueError or EOFError:
                print("Invalid Input\n")        


if num == 0:
        print(f"0 in words is: {Y}Zero{W}")
else:
        while num > 0:
                groupedList.append(num % 1000)
                num //= 1000

        groupedList.reverse()

        for i in groupedList:
                if i != 0:
                        if i >= 100:
                                name = name + numDict[int(i/100)] + " Hundred"
                                i = i % 100

                        if i >= 20:
                                if name == "":
                                        name = name + numDict[i  - (i % 10)]
                                else:
                                        name = name + " " + numDict[i - (i % 10)]

                                i = i % 10
                        elif i >= 10:
                                if name == "":
                                        name = name + numDict[i]
                                else:
                                        name = name + " " + numDict[i]

                                i = i % 10

                        if i != 0:
                                if name == "":
                                        name = name + numDict[i]
                                else:
                                        name = name + " " + numDict[i]

                        nameList.append(name)
                        name = ""
                else:
                        nameList.append("")

        for i in range(len(groupedList)):
                if nameList[i] != "":
                        name = name + nameList[i] + " " + placeValueDict[len(groupedList) - i] + " "

        name = name.rstrip()

        if len(splittedNum) == 2 and splittedNum[1] != "":
                name = name + f" {B}Point{Y}"

                for i in splittedNum[1]:
                        name = name + " " + digits[i]

                print(f"{W}", end="")

        if isNegative == False:
                print(f"\n{placeholder} in words is: {Y}{name}{W}")
        else:
                print(f"\n{placeholder} in words is: {Y}Minus {name}{W}")
