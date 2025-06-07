
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import uic

class MainWindow(QMainWindow):
   def __init__(self):
        super(MainWindow, self).__init__()
        
        # Load the UI file
        uic.loadUi('Emoji Dictionary/QT_GUI.ui', self)
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
            
        self.emoji_buttons = []
        self.emoji_layout = QGridLayout()
        self.emoji_widget = QWidget()
        self.emoji_widget.setLayout(self.emoji_layout)
        self.frame_2.setLayout(QVBoxLayout())
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
               #  button.clicked.connect(lambda checked, e=emoji: self.emoji_clicked(e))
                self.emoji_layout.addWidget(button, row_idx, col_idx)
                self.emoji_buttons.append(button)   
       
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
