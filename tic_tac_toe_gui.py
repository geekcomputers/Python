import tkinter as tk
from tkinter import messagebox

class TicTacToe:
    def __init__(self):
        # Create the main window
        self.window = tk.Tk()
        self.window.title("Tic Tac Toe - Python GUI")
        
        # X always starts
        self.turn = "X"
        self.buttons = []
        self.game_over = False

        # Create a 3x3 Grid of Buttons
        for i in range(3):
            row = []
            for j in range(3):
                # Create a single button
                btn = tk.Button(self.window, text="", font=("Arial", 20), width=5, height=2,
                                command=lambda x=i, y=j: self.on_click(x, y))
                # Place it on the grid
                btn.grid(row=i, column=j)
                row.append(btn)
            self.buttons.append(row)

        # Start the application loop
        self.window.mainloop()

    def on_click(self, row, col):
        # If the button is already clicked or game is over, do nothing
        if self.game_over or self.buttons[row][col]["text"] != "":
            return

        # Set the button text to X or O
        self.buttons[row][col]["text"] = self.turn
        
        # Check if someone won
        if self.check_winner():
            messagebox.showinfo("Game Over", f"Player {self.turn} wins!")
            self.game_over = True
        elif self.check_draw():
            messagebox.showinfo("Game Over", "It's a Draw!")
            self.game_over = True
        else:
            # Switch turns
            self.turn = "O" if self.turn == "X" else "X"

    def check_winner(self):
        # Check all rows, columns, and diagonals for a match
        for i in range(3):
            if self.buttons[i][0]["text"] == self.buttons[i][1]["text"] == self.buttons[i][2]["text"] != "":
                return True
            if self.buttons[0][i]["text"] == self.buttons[1][i]["text"] == self.buttons[2][i]["text"] != "":
                return True
        
        # Diagonals
        if self.buttons[0][0]["text"] == self.buttons[1][1]["text"] == self.buttons[2][2]["text"] != "":
            return True
        if self.buttons[0][2]["text"] == self.buttons[1][1]["text"] == self.buttons[2][0]["text"] != "":
            return True
            
        return False

    def check_draw(self):
        for row in self.buttons:
            for btn in row:
                if btn["text"] == "":
                    return False
        return True

if __name__ == "__main__":
    game = TicTacToe()
