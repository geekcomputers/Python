# cc program
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import *
import requests
from bs4 import BeautifulSoup
from requests.models import ContentDecodingError


def getVal(cont1, cont2):
    cont1val = cont1.split("-")[1]
    cont2val = cont2.split("-")[1]
    url = f"https://free.currconv.com/api/v7/convert?q={cont1val}_{cont2val}&compact=ultra&apiKey=b43a653672c4a94c4c26"
    r = requests.get(url)
    htmlContent = r.content
    soup = BeautifulSoup(htmlContent, "html.parser")
    try:
        valCurr = float(soup.get_text().split(":")[1].removesuffix("}"))  # {USD:70.00}
    except Exception:
        print("Server down.")
        exit()
    return valCurr


app = QtWidgets.QApplication([])

window = uic.loadUi("gui.ui")
f = open("country.txt", "r")

window = uic.loadUi("C:/Users/prath/Desktop/Currency-Calculator-Dynamic/gui.ui")
f = open("C:/Users/prath/Desktop/Currency-Calculator-Dynamic/country.txt", "r")

window.dropDown1.addItem("Select")
window.dropDown2.addItem("Select")
for i in f.readlines():
    window.dropDown1.addItem(i)
    window.dropDown2.addItem(i)
intOnly = QDoubleValidator()
window.lineEdit.setValidator(intOnly)


def main():
    window.pushButton.clicked.connect(changeBtn)


def changeBtn():
    val = window.lineEdit.text()
    cont1 = window.dropDown1.currentText()
    cont2 = window.dropDown2.currentText()
    valCurr = getVal(cont1.rstrip(), cont2.rstrip())
    window.lcdpanel.display(float(val) * valCurr)


main()
window.show()
app.exec()
