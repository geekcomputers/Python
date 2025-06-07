
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import uic
from emoji import demojize
import os

class MainWindow(QMainWindow):
   def __init__(self):
      super(MainWindow, self).__init__()
        
      # Load the UI file
      uic.loadUi(os.path.join(os.path.dirname(__file__),'QT_GUI.ui'),self)
      self.pushButton_4.clicked.connect(self.close)
      self.pushButton_2.clicked.connect(lambda:search_emoji())
      self.pushButton_3.clicked.connect(lambda:clear_text())
      cells = [
         
         ["🐒", "🐕", "🐎", "🐪", "🐁", "🐘", "🦘", "🦈", "🐓", "🐝", "👀", "🦴", "👩🏿", "‍🤝", "🧑", "🏾", "👱🏽", "‍♀", "🎞", "🎨", "⚽"],
         ["🍕", "🍗", "🍜", "☕", "🍴", "🍉", "🍓", "🌴", "🌵", "🛺", "🚲", "🛴", "🚉", "🚀", "✈", "🛰", "🚦", "🏳", "‍🌈", "🌎", "🧭"],
         ["🔥", "❄", "🌟", "🌞", "🌛", "🌝", "🌧", "🧺", "🧷", "🪒", "⛲", "🗼", "🕌", "👁", "‍🗨", "💬", "™", "💯", "🔕", "💥", "❤"],
         ["😀", "🥰", "😴", "🤓", "🤮", "🤬", "😨", "🤑", "😫", "😎"],
      ]
      def emoji_wight_btn():
         if self.emoji_widget.isVisible():
            self.emoji_widget.hide()
         else:
            self.emoji_widget.show() 
         
      def search_emoji():
            word = self.lineEdit.text()
            print(f"Field Text: {word}")           
            if word == "":
               self.textEdit.setText("You have entered no emoji.")
            else:
               means = demojize(word)
               self.textEdit.setText("Meaning of Emoji  :  " + str(word) + "\n\n" + means.replace("::", ":\n: "))
      
      def add_input_emoji(emoji):
         self.lineEdit.setText(self.lineEdit.text() + emoji)
      
      def clear_text():
         self.lineEdit.setText("")
         self.textEdit.setText("")
            
      self.emoji_buttons = []
      self.emoji_layout = QGridLayout()
      self.emoji_widget = QWidget()
      self.emoji_widget.setLayout(self.emoji_layout)
      self.frame_2.layout().addWidget(self.emoji_widget)
      self.emoji_widget.hide()
      self.pushButton.clicked.connect(lambda:emoji_wight_btn())
      
      
      for row_idx, row in enumerate(cells):
         for col_idx, emoji in enumerate(row):
               button = QPushButton(emoji)
               button.setFixedSize(40, 40)
               button.setFont(QFont("Arial", 20))
               button.setStyleSheet("""
                  QPushButton {
                     background-color: #ffffff;
                     border: 1px solid #e0e0e0;
                     border-radius: 5px;
                  }
                  QPushButton:hover {
                     background-color: #f0f0f0;
                  }
               """)
               button.clicked.connect(lambda checked, e=emoji: add_input_emoji(e))
               self.emoji_layout.addWidget(button, row_idx, col_idx)
               self.emoji_buttons.append(button)   
       
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
