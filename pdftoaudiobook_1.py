import pyttsx3
import pyPDF2
book = open('book.pdf','rb')
pdfreader = pyPDF2.PdfFileReader(book)
pages = pdfreader.numPages
print(pages)
speaker = pyttsx3.init()
page= pdfreader.getpage(7)
text = page.extractText()
speaker.say(text)
speaker.runAndWait()
