from tkinter import messagebox
import customtkinter as ctk
from typing import List, Tuple, Optional, Union, Literal
import time
import random
import logging

# Configure logging for error tracking
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("tictactoe.log"), logging.StreamHandler()]
)

# -------------------- Core Configuration --------------------
ctk.set_appearance_mode("System")  # Default to system appearance
ctk.set_default_color_theme("blue")  # Modern blue theme

# -------------------- Game Modes --------------------
MODE_EASY = "Easy"
MODE_HARD = "Hard"


class TicTacToe:
    """
    A modern Tic-Tac-Toe game with AI opponent, featuring:
    
    - Smooth animations for moves, wins, and resets
    - Dark/light mode toggle support
    - Adjustable difficulty (Easy/Hard)
    - Optimized AI with minimax algorithm
    - Robust error handling and boundary checks
    - Responsive, modern UI design
    
    The game follows standard Tic-Tac-Toe rules where the first player to get
    three of their marks in a row (horizontally, vertically, or diagonally) wins.
    Human player is 'X' and AI is 'O'.
    """

    def __init__(self, master: ctk.CTk) -> None:
        """
        Initialize the Tic-Tac-Toe game.
        
        Args:
            master: The parent Tkinter/CTkinter window
        """
        self.master = master
        self.master.title("Tic Tac Toe AI")
        self.master.geometry("500x650")
        self.master.resizable(False, False)
        
        # Game state initialization
        self.HUMAN: Literal["X"] = "X"
        self.AI: Literal["O"] = "O"
        self.EMPTY: Literal[" "] = " "
        self.BOARD_SIZE: int = 3  # 3x3 grid
        self.game_over: bool = False
        self.winning_cells: List[Tuple[int, int]] = []
        self.game_mode: str = MODE_HARD  # Default to hard mode
        
        # Precompute winning lines for faster checks
        self.winning_lines: List[List[Tuple[int, int]]] = self._get_winning_lines()

        # UI Style Configuration - Modern, cohesive design
        self.STYLES = {
            "empty": ("#F0F4F8", "#2D3748"),  # Light gray / Dark gray
            "human": ("#3182CE", "#4299E1"),   # Blue shades
            "ai": ("#E53E3E", "#F56565"),      # Red shades
            "winning": "#ED8936",              # Orange for winning line
            "draw": "#A0A0A0",                 # Gray for draw
            "button": {
                "font": ("Segoe UI", 40, "bold"),
                "width": 130,
                "height": 130,
                "corner_radius": 15,
                "hover_color": ("#E2E8F0", "#4A5568"),
                "text_color": ("#1A202C", "#F7FAFC"),
                "border_width": 2,
                "border_color": ("#CBD5E0", "#4A5568"),
            },
            "status": {
                "font": ("Segoe UI", 20, "bold"),
                "text_color": ("#1A202C", "#F7FAFC")
            },
            "mode": {
                "font": ("Segoe UI", 14, "bold"),
                "width": 130,
                "height": 40,
                "corner_radius": 10,
            },
            "info_button": {
                "font": ("Segoe UI", 14, "bold"),
                "width": 130,
                "height": 40,
                "corner_radius": 10,
            },
            "footer_button": {
                "font": ("Segoe UI", 14, "bold"),
                "width": 180,
                "height": 50,
                "corner_radius": 12,
            },
            "frame": {
                "corner_radius": 20,
                "fg_color": ("#FFFFFF", "#1A202C"),
                "border_width": 1,
                "border_color": ("#E2E8F0", "#4A5568"),
            },
        }

        # Initialize game board and UI components
        self._initialize_board()
        self._create_header_frame()
        self._create_board_widgets()
        self._create_footer_frame()

        logging.info("Tic-Tac-Toe game initialized successfully")

    def _initialize_board(self) -> None:
        """Initialize the game board matrix with empty cells"""
        self.board = [[self.EMPTY for _ in range(self.BOARD_SIZE)] 
                     for _ in range(self.BOARD_SIZE)]
        self.buttons: List[List[ctk.CTkButton]] = []  # Will hold the UI buttons for each cell

    def _get_winning_lines(self) -> List[List[Tuple[int, int]]]:
        """
        Precompute all possible winning lines (rows, columns, diagonals).
        
        Returns:
            List of lists containing tuples of (row, col) coordinates
            for each winning line configuration
        """
        lines: List[List[Tuple[int, int]]] = []
        # Rows
        lines.extend([[(i, j) for j in range(self.BOARD_SIZE)] 
                     for i in range(self.BOARD_SIZE)])
        # Columns
        lines.extend([[(j, i) for j in range(self.BOARD_SIZE)] 
                     for i in range(self.BOARD_SIZE)])
        # Diagonals
        lines.append([(i, i) for i in range(self.BOARD_SIZE)])
        lines.append([(i, self.BOARD_SIZE - 1 - i) 
                     for i in range(self.BOARD_SIZE)])
        return lines

    def _create_header_frame(self) -> None:
        """Create header frame with game status and control buttons"""
        try:
            header_frame = ctk.CTkFrame(self.master, **self.STYLES["frame"])
            header_frame.pack(fill="x", padx=20, pady=15, ipady=10)
            
            # Configure grid for header layout
            header_frame.grid_columnconfigure(0, weight=1)
            header_frame.grid_columnconfigure(1, weight=0)
            
            # Game status display
            self.status_label = ctk.CTkLabel(
                header_frame,
                text="Your Turn (X)",
                **self.STYLES["status"],
            )
            self.status_label.grid(row=0, column=0, padx=20, pady=10, sticky="w")
            
            # Control buttons frame
            control_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
            control_frame.grid(row=0, column=1, padx=10, pady=5)
            
            # Difficulty mode toggle
            self.mode_button = ctk.CTkButton(
                control_frame,
                text=f"Mode: {self.game_mode}",
                **self.STYLES["mode"],
                command=self._toggle_mode,
                fg_color=("#4299E1", "#3182CE"),
            )
            self.mode_button.pack(side="left", padx=(0, 10))
            
            # How to play button
            self.info_button = ctk.CTkButton(
                control_frame,
                text="How to Play",
                **self.STYLES["info_button"],
                command=self._show_instructions,
                fg_color=("#38B2AC", "#319795"),
            )
            self.info_button.pack(side="left")
        except Exception as e:
            logging.error(f"Error creating header frame: {str(e)}", exc_info=True)
            messagebox.showerror("UI Error", "Failed to create header: " + str(e))

    def _create_board_widgets(self) -> None:
        """Create 3x3 game board with interactive buttons"""
        try:
            board_frame = ctk.CTkFrame(self.master, **self.STYLES["frame"])
            board_frame.pack(fill="both", expand=True, padx=20, pady=5)
            
            # Configure grid weights for responsive layout
            for i in range(self.BOARD_SIZE):
                board_frame.grid_rowconfigure(i, weight=1)
                board_frame.grid_columnconfigure(i, weight=1)

            # Create buttons for each cell
            for i in range(self.BOARD_SIZE):
                row_buttons: List[ctk.CTkButton] = []
                for j in range(self.BOARD_SIZE):
                    button = ctk.CTkButton(
                        board_frame,
                        text=self.EMPTY,
                        fg_color=self.STYLES["empty"],
                        **self.STYLES["button"],
                        command=lambda r=i, c=j: self._handle_cell_click(r, c),
                    )
                    button.grid(row=i, column=j, padx=8, pady=8, sticky="nsew")
                    row_buttons.append(button)
                self.buttons.append(row_buttons)
        except Exception as e:
            logging.error(f"Error creating board widgets: {str(e)}", exc_info=True)
            messagebox.showerror("UI Error", "Failed to create game board: " + str(e))

    def _create_footer_frame(self) -> None:
        """Create footer frame with theme toggle and new game button"""
        try:
            footer_frame = ctk.CTkFrame(self.master, **self.STYLES["frame"])
            footer_frame.pack(fill="x", padx=20, pady=15, ipady=10)
            
            # Footer layout with centered buttons
            footer_container = ctk.CTkFrame(footer_frame, fg_color="transparent")
            footer_container.pack(expand=True, pady=5)
            
            # Dark/light mode toggle
            self.theme_button = ctk.CTkButton(
                footer_container,
                text="Toggle Dark Mode",
                **self.STYLES["footer_button"],
                command=self._toggle_theme,
                fg_color=("#718096", "#4A5568"),
            )
            self.theme_button.pack(side="left", padx=(0, 15))
            
            # New game button
            self.new_game_button = ctk.CTkButton(
                footer_container,
                text="New Game",
                **self.STYLES["footer_button"],
                command=self._reset_game,
                fg_color=("#38B2AC", "#319795"),
                hover_color=("#319795", "#2C7A7B"),
            )
            self.new_game_button.pack(side="left")
        except Exception as e:
            logging.error(f"Error creating footer frame: {str(e)}", exc_info=True)
            messagebox.showerror("UI Error", "Failed to create footer: " + str(e))

    def _toggle_theme(self) -> None:
        """Toggle between light and dark mode"""
        try:
            current_mode = ctk.get_appearance_mode()
            new_mode = "Dark" if current_mode == "Light" else "Light"
            ctk.set_appearance_mode(new_mode)
            self.theme_button.configure(text=f"Toggle {current_mode} Mode")
            logging.info(f"Theme toggled to {new_mode} mode")
        except Exception as e:
            logging.error(f"Error toggling theme: {str(e)}", exc_info=True)
            messagebox.showerror("Theme Error", "Failed to toggle theme: " + str(e))

    def _toggle_mode(self) -> None:
        """Toggle between Easy and Hard difficulty modes"""
        try:
            self.game_mode = MODE_EASY if self.game_mode == MODE_HARD else MODE_HARD
            self.mode_button.configure(text=f"Mode: {self.game_mode}")
            messagebox.showinfo("Mode Changed", f"Switched to {self.game_mode} Mode")
            logging.info(f"Game mode changed to {self.game_mode}")
        except Exception as e:
            logging.error(f"Error toggling game mode: {str(e)}", exc_info=True)
            messagebox.showerror("Mode Error", "Failed to toggle mode: " + str(e))

    def _show_instructions(self) -> None:
        """Show game instructions in a dialog box"""
        try:
            instructions = """
            Tic Tac Toe Instructions:
            
            1. The game is played on a 3x3 grid.
            2. You are 'X' and the AI is 'O'.
            3. Players take turns placing their marks in empty squares.
            4. The first player to get 3 of their marks in a row 
               (horizontally, vertically, or diagonally) wins.
            5. If all squares are filled with no winner, the game is a draw.
            
            Modes:
            - Easy: AI makes occasional mistakes (40% chance)
            - Hard: AI plays optimally (unbeatable)
            """
            messagebox.showinfo("How to Play", instructions)
        except Exception as e:
            logging.error(f"Error showing instructions: {str(e)}", exc_info=True)
            messagebox.showerror("Info Error", "Failed to show instructions: " + str(e))

    def _handle_cell_click(self, row: int, col: int) -> None:
        """
        Handle human player's cell click.
        
        Args:
            row: Row index of the clicked cell (0-2)
            col: Column index of the clicked cell (0-2)
        """
        try:
            # Validate input coordinates
            if not (0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE):
                logging.warning(f"Invalid cell click: row={row}, col={col}")
                return

            # Check if game is over or cell is already occupied
            if self.game_over or self.board[row][col] != self.EMPTY:
                return

            # Human move with animation
            self.board[row][col] = self.HUMAN
            self._animate_move(row, col, self.HUMAN)

            # Check game state after human move
            if self._check_winner(self.HUMAN):
                self._end_game(winner="human")
                return

            if self._is_board_full():
                self._end_game(winner=None)
                return

            # AI's turn with short delay for better UX
            self.status_label.configure(text="AI is thinking...")
            self.master.update()  # Force UI update
            self.master.after(500, self._make_ai_move)  # 500ms delay

        except Exception as e:
            logging.error(f"Error handling cell click: {str(e)}", exc_info=True)
            messagebox.showerror("Game Error", "Error processing move: " + str(e))

    def _animate_move(self, row: int, col: int, player: Literal["X", "O"]) -> None:
        """
        Animate a move with color transition effect.
        
        Args:
            row: Row index of the cell (0-2)
            col: Column index of the cell (0-2)
            player: Player making the move ("X" for human, "O" for AI)
        """
        try:
            # Validate coordinates
            if not (0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE):
                raise ValueError(f"Invalid cell coordinates: ({row}, {col})")

            target_color = self.STYLES["human"] if player == self.HUMAN else self.STYLES["ai"]
            start_color = self.STYLES["empty"]
            
            # Animate color transition over 20 steps
            for i in range(21):
                current_color = self._interpolate_color(start_color, target_color, i/20)
                self.buttons[row][col].configure(
                    text=player,
                    fg_color=current_color
                )
                self.master.update()
                time.sleep(0.01)  # Short delay for smooth animation

        except Exception as e:
            logging.error(f"Error animating move: {str(e)}", exc_info=True)

    def _interpolate_color(self, start: Union[str, Tuple[str, str]], 
                          target: Union[str, Tuple[str, str]], 
                          factor: float) -> str:
        """
        Interpolate between two colors, handling both string and tuple formats.
        
        Args:
            start: Starting color (either hex string or (light, dark) tuple)
            target: Target color (either hex string or (light, dark) tuple)
            factor: Interpolation factor (0.0 to 1.0)
            
        Returns:
            Hex string representing the interpolated color
        """
        try:
            # Determine current theme (light=0, dark=1)
            theme_idx = 0 if ctk.get_appearance_mode() == "Light" else 1
            
            # Extract colors based on format
            if isinstance(start, tuple):
                start_color = start[theme_idx]
            else:
                start_color = start
                
            if isinstance(target, tuple):
                target_color = target[theme_idx]
            else:
                target_color = target
                
            # Convert hex to RGB
            def hex_to_rgb(h: str) -> Tuple[int, int, int]:
                h = h.lstrip('#')
                if len(h) != 6:
                    raise ValueError(f"Invalid hex color: {h}")
                return (
                    int(h[0:2], 16),
                    int(h[2:4], 16),
                    int(h[4:6], 16)
                )
                
            # Interpolate RGB values
            s_r, s_g, s_b = hex_to_rgb(start_color)
            t_r, t_g, t_b = hex_to_rgb(target_color)
            
            # Clamp factor between 0 and 1
            factor = max(0.0, min(1.0, factor))
            
            r = int(s_r + (t_r - s_r) * factor)
            g = int(s_g + (t_g - s_g) * factor)
            b = int(s_b + (t_b - s_b) * factor)
            
            # Convert back to hex
            return f'#{r:02x}{g:02x}{b:02x}'
            
        except Exception as e:
            logging.error(f"Error interpolating color: {str(e)}", exc_info=True)
            # Return safe default color
            return "#FFFFFF" if ctk.get_appearance_mode() == "Light" else "#000000"

    def _make_ai_move(self) -> None:
        """Make AI move with strategy based on difficulty mode"""
        try:
            if self.game_over:  # Double-check game state
                return

            best_move = self._find_best_move()
            if best_move is not None:
                row, col = best_move
                
                # Validate move coordinates
                if not (0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE):
                    raise ValueError(f"AI calculated invalid move: ({row}, {col})")

                # Make AI move with animation
                self.board[row][col] = self.AI
                self._animate_move(row, col, self.AI)

                # Check game state after AI move
                if self._check_winner(self.AI):
                    self._end_game(winner="ai")
                    return

                if self._is_board_full():
                    self._end_game(winner=None)
                    return

            # Update status after valid move
            self.status_label.configure(text="Your Turn (X)")

        except Exception as e:
            logging.error(f"Error making AI move: {str(e)}", exc_info=True)
            self.status_label.configure(text="Error: AI move failed")
            messagebox.showerror("AI Error", "AI failed to make a move: " + str(e))

    def _check_winner(self, player: Literal["X", "O"]) -> bool:
        """
        Check if the specified player has won.
        
        Args:
            player: Player to check ("X" for human, "O" for AI)
            
        Returns:
            True if player has a winning line, False otherwise
        """
        try:
            if player not in (self.HUMAN, self.AI):
                raise ValueError(f"Invalid player: {player}")

            # Check all winning lines
            for line in self.winning_lines:
                if all(self.board[i][j] == player for i, j in line):
                    self.winning_cells = line  # Store winning cells for animation
                    return True
            return False
        except Exception as e:
            logging.error(f"Error checking winner: {str(e)}", exc_info=True)
            return False

    def _is_board_full(self) -> bool:
        """
        Check if the board has no empty cells.
        
        Returns:
            True if board is full, False otherwise
        """
        try:
            # Check if all cells are filled
            for i in range(self.BOARD_SIZE):
                for j in range(self.BOARD_SIZE):
                    if self.board[i][j] == self.EMPTY:
                        return False
            return True
        except Exception as e:
            logging.error(f"Error checking board fullness: {str(e)}", exc_info=True)
            return False  # Safe default

    def _end_game(self, winner: Optional[Literal["human", "ai"]]) -> None:
        """
        End the game and handle game over state.
        
        Args:
            winner: "human" if human won, "ai" if AI won, None for draw
        """
        try:
            if self.game_over:  # Prevent duplicate calls
                return
                
            self.game_over = True
            
            # Animate winning cells or draw
            if winner:
                # Highlight winning line with animation
                for i, j in self.winning_cells:
                    self._animate_winning_cell(i, j)
                
                # Update status and show message
                if winner == "human":
                    text = "You Win!"
                    logging.info("Human player won the game")
                else:
                    text = "AI Wins!"
                    logging.info("AI player won the game")
                    
                self.status_label.configure(text=text)
                messagebox.showinfo("Game Over", text)
                
            else:
                # Draw - animate all cells
                for i in range(self.BOARD_SIZE):
                    for j in range(self.BOARD_SIZE):
                        self._animate_draw_cell(i, j)
                
                text = "It's a Draw!"
                self.status_label.configure(text=text)
                messagebox.showinfo("Game Over", text)
                logging.info("Game ended in a draw")

        except Exception as e:
            logging.error(f"Error ending game: {str(e)}", exc_info=True)
            messagebox.showerror("Game Over Error", "Failed to end game properly: " + str(e))

    def _animate_winning_cell(self, row: int, col: int) -> None:
        """
        Animate winning cells with a pulse effect.
        
        Args:
            row: Row index of the winning cell (0-2)
            col: Column index of the winning cell (0-2)
        """
        try:
            # Validate coordinates
            if not (0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE):
                raise ValueError(f"Invalid winning cell: ({row}, {col})")

            # Get current color based on theme
            theme_idx = 0 if ctk.get_appearance_mode() == "Light" else 1
            original_color = (self.STYLES["human"][theme_idx] 
                            if self.board[row][col] == self.HUMAN 
                            else self.STYLES["ai"][theme_idx])
            
            # Pulse animation - 3 cycles of brightening and dimming
            for _ in range(3):
                # Brighten
                for i in range(11):
                    factor = 1 + i * 0.1
                    self._adjust_button_brightness(row, col, original_color, factor)
                    self.master.update()
                    time.sleep(0.03)
                
                # Dim back
                for i in range(10, -1, -1):
                    factor = 1 + i * 0.1
                    self._adjust_button_brightness(row, col, original_color, factor)
                    self.master.update()
                    time.sleep(0.03)
            
            # Set to final winning color
            self.buttons[row][col].configure(fg_color=self.STYLES["winning"])

        except Exception as e:
            logging.error(f"Error animating winning cell: {str(e)}", exc_info=True)

    def _animate_draw_cell(self, row: int, col: int) -> None:
        """
        Animate draw cells with color transition to gray.
        
        Args:
            row: Row index of the cell (0-2)
            col: Column index of the cell (0-2)
        """
        try:
            # Validate coordinates
            if not (0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE):
                raise ValueError(f"Invalid draw cell: ({row}, {col})")

            # Get current color based on theme
            theme_idx = 0 if ctk.get_appearance_mode() == "Light" else 1
            original_color = (self.STYLES["human"][theme_idx] 
                            if self.board[row][col] == self.HUMAN 
                            else self.STYLES["ai"][theme_idx])
            
            # Transition to draw color
            draw_color = self.STYLES["draw"]
            
            for i in range(21):
                factor = i / 20
                current_color = self._interpolate_color(original_color, draw_color, factor)
                self.buttons[row][col].configure(fg_color=current_color)
                self.master.update()
                time.sleep(0.02)

        except Exception as e:
            logging.error(f"Error animating draw cell: {str(e)}", exc_info=True)

    def _adjust_button_brightness(self, row: int, col: int, 
                                base_color: Union[str, Tuple[str, str]], 
                                factor: float) -> None:
        """
        Adjust button brightness for animation effects.
        
        Args:
            row: Row index of the cell (0-2)
            col: Column index of the cell (0-2)
            base_color: Original color to adjust
            factor: Brightness factor (>1 brightens, <1 darkens)
        """
        try:
            # Validate coordinates
            if not (0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE):
                raise ValueError(f"Invalid cell for brightness adjustment: ({row}, {col})")

            # Determine current theme
            theme_idx = 0 if ctk.get_appearance_mode() == "Light" else 1
            
            # Extract color based on format
            if isinstance(base_color, tuple):
                color = base_color[theme_idx]
            else:
                color = base_color
                
            # Convert hex to RGB
            def hex_to_rgb(h: str) -> Tuple[int, int, int]:
                h = h.lstrip('#')
                if len(h) != 6:
                    raise ValueError(f"Invalid hex color: {h}")
                return (
                    int(h[0:2], 16),
                    int(h[2:4], 16),
                    int(h[4:6], 16)
                )
                
            # Convert RGB to hex
            def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
                return '#%02x%02x%02x' % rgb
                
            r, g, b = hex_to_rgb(color)
            
            # Adjust brightness with clamping
            def clamp(x: float) -> int:
                return max(0, min(255, int(x)))
                
            new_r = clamp(r * factor)
            new_g = clamp(g * factor)
            new_b = clamp(b * factor)
            
            # Update button color
            self.buttons[row][col].configure(fg_color=rgb_to_hex((new_r, new_g, new_b)))

        except Exception as e:
            logging.error(f"Error adjusting brightness: {str(e)}", exc_info=True)

    def _find_best_move(self) -> Optional[Tuple[int, int]]:
        """
        Find optimal move for AI based on difficulty mode.
        
        Returns:
            Tuple of (row, col) for best move, or None if no moves available
        """
        try:
            # Check for immediate win opportunity
            for i in range(self.BOARD_SIZE):
                for j in range(self.BOARD_SIZE):
                    if self.board[i][j] == self.EMPTY:
                        self.board[i][j] = self.AI  # Try the move
                        if self._check_winner(self.AI):
                            self.board[i][j] = self.EMPTY  # Undo
                            return (i, j)  # Return winning move
                        self.board[i][j] = self.EMPTY  # Undo

            # Check for human's immediate win to block
            for i in range(self.BOARD_SIZE):
                for j in range(self.BOARD_SIZE):
                    if self.board[i][j] == self.EMPTY:
                        self.board[i][j] = self.HUMAN  # Try human's potential move
                        if self._check_winner(self.HUMAN):
                            self.board[i][j] = self.EMPTY  # Undo
                            return (i, j)  # Block this move
                        self.board[i][j] = self.EMPTY  # Undo

            # Different strategies based on difficulty
            if self.game_mode == MODE_EASY:
                # 40% chance of random move in easy mode
                if random.random() < 0.4:
                    random_move = self._get_random_move()
                    if random_move:
                        return random_move
            
            # Hard mode uses minimax algorithm for optimal play
            return self._get_minimax_move()

        except Exception as e:
            logging.error(f"Error finding best move: {str(e)}", exc_info=True)
            return self._get_random_move()  # Fallback to random move

    def _get_random_move(self) -> Optional[Tuple[int, int]]:
        """
        Get a random valid move (for easy mode or error fallback).
        
        Returns:
            Tuple of (row, col) for random move, or None if no moves available
        """
        try:
            available_moves = [(i, j) for i in range(self.BOARD_SIZE) 
                             for j in range(self.BOARD_SIZE) 
                             if self.board[i][j] == self.EMPTY]
            
            if available_moves:
                return random.choice(available_moves)
            return None
        except Exception as e:
            logging.error(f"Error getting random move: {str(e)}", exc_info=True)
            return None

    def _get_minimax_move(self) -> Optional[Tuple[int, int]]:
        """
        Find optimal move using minimax algorithm with alpha-beta pruning.
        
        Returns:
            Tuple of (row, col) for best move, or None if no moves available
        """
        try:
            best_score = float('-inf')
            best_move: Optional[Tuple[int, int]] = None
            depth_limit = 9  # Full depth for 3x3 board

            # Prefer strategic positions (center, corners, edges)
            preferred_moves = [(1, 1)]  # Center first
            preferred_moves.extend([(0,0), (0,2), (2,0), (2,2)])  # Corners
            preferred_moves.extend([(0,1), (1,0), (1,2), (2,1)])  # Edges

            for i, j in preferred_moves:
                if self.board[i][j] == self.EMPTY:
                    self.board[i][j] = self.AI  # Try the move
                    # Evaluate with minimax
                    score = self._minimax(0, False, depth_limit, float('-inf'), float('inf'))
                    self.board[i][j] = self.EMPTY  # Undo
                    
                    if score > best_score:
                        best_score = score
                        best_move = (i, j)

            return best_move
        except Exception as e:
            logging.error(f"Error in minimax calculation: {str(e)}", exc_info=True)
            return self._get_random_move()  # Fallback to random move

    def _minimax(self, depth: int, is_maximizing: bool, depth_limit: int, 
                alpha: float, beta: float) -> int:
        """
        Minimax algorithm with alpha-beta pruning for optimal play.
        
        Args:
            depth: Current recursion depth
            is_maximizing: True if AI's turn (maximizing), False if human's (minimizing)
            depth_limit: Maximum recursion depth to prevent excessive computation
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            
        Returns:
            Score evaluation of the current board state
        """
        try:
            # Base cases
            if self._check_winner(self.AI):
                return 10 - depth  # Positive score for AI win (higher for faster wins)
            elif self._check_winner(self.HUMAN):
                return -10 + depth  # Negative score for human win (less negative for slower wins)
            elif self._is_board_full() or depth >= depth_limit:
                return 0  # Neutral score for draw or depth limit

            if is_maximizing:
                # AI's turn - maximize score
                max_score = float('-inf')
                for i in range(self.BOARD_SIZE):
                    for j in range(self.BOARD_SIZE):
                        if self.board[i][j] == self.EMPTY:
                            self.board[i][j] = self.AI  # Try move
                            score = self._minimax(depth + 1, False, depth_limit, alpha, beta)
                            self.board[i][j] = self.EMPTY  # Undo
                            max_score = max(score, max_score)
                            alpha = max(alpha, score)
                            if beta <= alpha:
                                break  # Prune this branch
                    if beta <= alpha:
                        break  # Prune this branch
                return max_score
            else:
                # Human's turn - minimize score
                min_score = float('inf')
                for i in range(self.BOARD_SIZE):
                    for j in range(self.BOARD_SIZE):
                        if self.board[i][j] == self.EMPTY:
                            self.board[i][j] = self.HUMAN  # Try human's move
                            score = self._minimax(depth + 1, True, depth_limit, alpha, beta)
                            self.board[i][j] = self.EMPTY  # Undo
                            min_score = min(score, min_score)
                            beta = min(beta, score)
                            if beta <= alpha:
                                break  # Prune this branch
                    if beta <= alpha:
                        break  # Prune this branch
                return min_score
        except Exception as e:
            logging.error(f"Error in minimax recursion: {str(e)}", exc_info=True)
            return 0  # Neutral score on error

    def _reset_game(self) -> None:
        """Reset game to initial state with animation"""
        try:
            # Reset game state but keep button references
            self.board = [[self.EMPTY for _ in range(self.BOARD_SIZE)] 
                         for _ in range(self.BOARD_SIZE)]
            self.game_over = False
            self.winning_cells = []
            
            # Animate reset for all cells
            for i in range(self.BOARD_SIZE):
                for j in range(self.BOARD_SIZE):
                    self._animate_reset_cell(i, j)
            
            # Reset status
            self.status_label.configure(text="Your Turn (X)")
            logging.info("Game reset successfully")
        except Exception as e:
            logging.error(f"Error resetting game: {str(e)}", exc_info=True)
            messagebox.showerror("Reset Error", "Failed to reset game: " + str(e))

    def _animate_reset_cell(self, row: int, col: int) -> None:
        """
        Animate cell reset with fade effect.
        
        Args:
            row: Row index of the cell (0-2)
            col: Column index of the cell (0-2)
        """
        try:
            # Validate coordinates and button existence
            if not (0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE):
                raise ValueError(f"Invalid cell for reset animation: ({row}, {col})")
                
            if not (0 <= row < len(self.buttons) and 0 <= col < len(self.buttons[row])):
                logging.warning(f"Button not found at ({row}, {col}) during reset animation")
                return  # Skip animation for missing button
                
            # Get current color based on theme
            theme_idx = 0 if ctk.get_appearance_mode() == "Light" else 1
            current_color = self.buttons[row][col].cget("fg_color")
            
            if isinstance(current_color, tuple):
                current_color = current_color[theme_idx]
                
            target_color = self.STYLES["empty"]
            
            # Fade out over 15 steps
            for i in range(16):
                factor = 1 - (i / 15)
                self._adjust_button_brightness(row, col, current_color, factor)
                self.master.update()
                time.sleep(0.01)
            
            # Reset to empty state
            self.buttons[row][col].configure(
                text=self.EMPTY,
                fg_color=target_color
            )
        except Exception as e:
            logging.error(f"Error in reset animation: {str(e)}", exc_info=True)


if __name__ == "__main__":
    try:
        root = ctk.CTk()
        app = TicTacToe(root)
        root.mainloop()
    except Exception as e:
        logging.critical(f"Fatal error in main application: {str(e)}", exc_info=True)
        messagebox.showerror("Fatal Error", "The application could not start: " + str(e))