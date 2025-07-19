#!/usr/bin/env python3

import sqlite3
import tkinter as tk
from tkinter import Button, Entry, Frame, Label, Text, messagebox, ttk


class NotepadApp:
    def __init__(self, root: tk.Tk):
        """Initialize the Notepad Application"""
        self.root = root
        self.root.title("Advanced Notepad")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        # Ensure proper font for English display
        self.root.option_add("*Font", "Arial 10")

        # Database connection
        self.connection = sqlite3.connect("notes.db")
        self.cursor = self.connection.cursor()
        self._initialize_database()

        # Search state
        self.search_active = False
        self.current_results: list[tuple[int, str, str]] = []
        self.current_index = 0

        # Create UI
        self._create_ui()

    def _initialize_database(self) -> None:
        """Initialize the database table"""
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.connection.commit()
        except Exception as e:
            messagebox.showerror(
                "Database Error", f"Failed to initialize database: {str(e)}"
            )

    def _create_ui(self) -> None:
        """Create the user interface"""
        # Create main frame
        main_frame = Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create notebook tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Add tabs
        self.tab_add = Frame(self.notebook)
        self.tab_display = Frame(self.notebook)
        self.tab_search = Frame(self.notebook)

        self.notebook.add(self.tab_add, text="Add Note")
        self.notebook.add(self.tab_display, text="Browse Notes")
        self.notebook.add(self.tab_search, text="Search Notes")

        # Setup Add tab
        self._setup_add_tab()

        # Setup Display tab
        self._setup_display_tab()

        # Setup Search tab
        self._setup_search_tab()

        # Status bar
        self.status_bar = Label(
            self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Load all notes initially
        self._load_all_notes()

    def _setup_add_tab(self) -> None:
        """Setup the Add Note tab"""
        # Title frame
        title_frame = Frame(self.tab_add)
        title_frame.pack(fill=tk.X, padx=10, pady=5)

        Label(title_frame, text="Title:").pack(side=tk.LEFT, padx=5)
        self.title_entry = Entry(title_frame, width=60)
        self.title_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Content frame
        content_frame = Frame(self.tab_add)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        Label(content_frame, text="Content:").pack(anchor=tk.W, padx=5)
        self.content_text = Text(content_frame, wrap=tk.WORD, height=15)
        self.content_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Button frame
        button_frame = Frame(self.tab_add)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        Button(button_frame, text="Save Note", command=self._save_note).pack(
            side=tk.RIGHT, padx=5
        )
        Button(button_frame, text="Clear", command=self._clear_form).pack(
            side=tk.RIGHT, padx=5
        )

    def _setup_display_tab(self) -> None:
        """Setup the Browse Notes tab"""
        # Search frame
        search_frame = Frame(self.tab_display)
        search_frame.pack(fill=tk.X, padx=10, pady=5)

        Label(search_frame, text="Filter:").pack(side=tk.LEFT, padx=5)
        self.filter_entry = Entry(search_frame, width=40)
        self.filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        Button(search_frame, text="Apply Filter", command=self._apply_filter).pack(
            side=tk.RIGHT, padx=5
        )

        # Notes list frame
        list_frame = Frame(self.tab_display)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Treeview for notes
        columns = ("id", "title", "created_at")
        self.notes_tree = ttk.Treeview(list_frame, columns=columns, show="headings")
        self.notes_tree.heading("id", text="ID")
        self.notes_tree.heading("title", text="Title")
        self.notes_tree.heading("created_at", text="Created At")

        self.notes_tree.column("id", width=50)
        self.notes_tree.column("title", width=300)
        self.notes_tree.column("created_at", width=150)

        self.notes_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.notes_tree.yview
        )
        self.notes_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind selection event
        self.notes_tree.bind("<<TreeviewSelect>>", self._on_note_select)

        # Button frame
        button_frame = Frame(self.tab_display)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        Button(button_frame, text="Refresh", command=self._load_all_notes).pack(
            side=tk.LEFT, padx=5
        )
        Button(
            button_frame, text="Delete Selected", command=self._delete_selected_note
        ).pack(side=tk.RIGHT, padx=5)

    def _setup_search_tab(self) -> None:
        """Setup the Search Notes tab"""
        # Search frame
        search_frame = Frame(self.tab_search)
        search_frame.pack(fill=tk.X, padx=10, pady=5)

        Label(search_frame, text="Search Title:").pack(side=tk.LEFT, padx=5)
        self.search_entry = Entry(search_frame, width=50)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        Button(search_frame, text="Search", command=self._search_notes).pack(
            side=tk.RIGHT, padx=5
        )

        # Results frame
        results_frame = Frame(self.tab_search)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Navigation buttons
        nav_frame = Frame(results_frame)
        nav_frame.pack(fill=tk.X)

        self.prev_button = Button(
            nav_frame,
            text="Previous",
            command=self._show_previous_result,
            state=tk.DISABLED,
        )
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = Button(
            nav_frame, text="Next", command=self._show_next_result, state=tk.DISABLED
        )
        self.next_button.pack(side=tk.RIGHT, padx=5)

        # Result display
        self.result_title_var = tk.StringVar()
        Label(
            results_frame,
            textvariable=self.result_title_var,
            font=("Arial", 12, "bold"),
        ).pack(anchor=tk.W, padx=5, pady=5)

        self.result_content_text = Text(results_frame, wrap=tk.WORD, height=15)
        self.result_content_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Status label
        self.search_status_var = tk.StringVar()
        self.search_status_var.set("No search results")
        Label(results_frame, textvariable=self.search_status_var).pack(
            anchor=tk.W, padx=5, pady=5
        )

    def _load_all_notes(self) -> None:
        """Load all notes into the treeview"""
        # Clear existing items
        for item in self.notes_tree.get_children():
            self.notes_tree.delete(item)

        try:
            self.cursor.execute(
                "SELECT id, title, created_at FROM notes ORDER BY created_at DESC"
            )
            notes = self.cursor.fetchall()

            for note in notes:
                self.notes_tree.insert("", tk.END, values=note)

            self.status_bar.config(text=f"Loaded {len(notes)} notes")
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to load notes: {str(e)}")

    def _apply_filter(self) -> None:
        """Apply filter to notes list"""
        filter_text = self.filter_entry.get().strip().lower()

        # Clear existing items
        for item in self.notes_tree.get_children():
            self.notes_tree.delete(item)

        try:
            if filter_text:
                self.cursor.execute(
                    "SELECT id, title, created_at FROM notes WHERE LOWER(title) LIKE ? ORDER BY created_at DESC",
                    (f"%{filter_text}%",),
                )
            else:
                self.cursor.execute(
                    "SELECT id, title, created_at FROM notes ORDER BY created_at DESC"
                )

            notes = self.cursor.fetchall()

            for note in notes:
                self.notes_tree.insert("", tk.END, values=note)

            self.status_bar.config(text=f"Filtered {len(notes)} notes")
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to filter notes: {str(e)}")

    def _on_note_select(self, event: tk.Event) -> None:
        """Handle note selection"""
        selected_items = self.notes_tree.selection()
        if not selected_items:
            return

        note_id = self.notes_tree.item(selected_items[0])["values"][0]

        try:
            self.cursor.execute(
                "SELECT title, content FROM notes WHERE id = ?", (note_id,)
            )
            note = self.cursor.fetchone()

            if note:
                title, content = note
                # Open a new window to display the note
                self._open_note_viewer(note_id, title, content)
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to load note: {str(e)}")

    def _open_note_viewer(self, note_id: int, title: str, content: str) -> None:
        """Open a new window to view/edit the note"""
        viewer = tk.Toplevel(self.root)
        viewer.title(f"Note: {title}")
        viewer.geometry("600x500")
        viewer.resizable(True, True)

        # Title entry
        title_frame = Frame(viewer)
        title_frame.pack(fill=tk.X, padx=10, pady=5)

        title_label = Label(title_frame, text="Title:")
        title_label.pack(side=tk.LEFT, padx=5)

        title_entry = Entry(title_frame, width=50)
        title_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        title_entry.insert(0, title)

        # Content text
        content_text = Text(viewer, wrap=tk.WORD, height=20)
        content_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        content_text.insert(tk.END, content)

        # Button frame
        button_frame = Frame(viewer)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        def update_note() -> None:
            new_title = title_entry.get().strip()
            new_content = content_text.get(1.0, tk.END).strip()

            if not new_title:
                messagebox.showerror("Input Error", "Title cannot be empty")
                return

            try:
                self.cursor.execute(
                    "UPDATE notes SET title = ?, content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (new_title, new_content, note_id),
                )
                self.connection.commit()
                messagebox.showinfo("Success", "Note updated successfully")
                viewer.destroy()
                self._load_all_notes()
            except Exception as e:
                messagebox.showerror(
                    "Database Error", f"Failed to update note: {str(e)}"
                )

        Button(button_frame, text="Update Note", command=update_note).pack(
            side=tk.RIGHT, padx=5
        )
        Button(button_frame, text="Cancel", command=viewer.destroy).pack(
            side=tk.RIGHT, padx=5
        )

    def _delete_selected_note(self) -> None:
        """Delete the selected note"""
        selected_items = self.notes_tree.selection()
        if not selected_items:
            messagebox.showinfo("Selection Error", "Please select a note to delete")
            return

        note_id = self.notes_tree.item(selected_items[0])["values"][0]
        title = self.notes_tree.item(selected_items[0])["values"][1]

        if messagebox.askyesno(
            "Confirm Delete", f"Are you sure you want to delete the note '{title}'?"
        ):
            try:
                self.cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
                self.connection.commit()
                self._load_all_notes()
                self.status_bar.config(text=f"Deleted note with ID {note_id}")
            except Exception as e:
                messagebox.showerror(
                    "Database Error", f"Failed to delete note: {str(e)}"
                )

    def _save_note(self) -> None:
        """Save the current note to the database"""
        title = self.title_entry.get().strip()
        content = self.content_text.get(1.0, tk.END).strip()

        if not title:
            messagebox.showerror("Input Error", "Title cannot be empty")
            return

        try:
            self.cursor.execute(
                "INSERT INTO notes (title, content) VALUES (?, ?)", (title, content)
            )
            self.connection.commit()
            messagebox.showinfo("Success", "Note saved successfully")
            self._clear_form()
            self._load_all_notes()
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to save note: {str(e)}")

    def _clear_form(self) -> None:
        """Clear the form fields"""
        self.title_entry.delete(0, tk.END)
        self.content_text.delete(1.0, tk.END)
        self.status_bar.config(text="Form cleared")

    def _search_notes(self) -> None:
        """Search notes by title"""
        search_text = self.search_entry.get().strip()

        if not search_text:
            messagebox.showinfo("Search Error", "Please enter a search term")
            return

        try:
            self.cursor.execute(
                "SELECT id, title, content FROM notes WHERE title LIKE ? ORDER BY created_at DESC",
                (f"%{search_text}%",),
            )
            self.current_results = self.cursor.fetchall()
            self.current_index = 0

            if not self.current_results:
                self.search_status_var.set("No results found")
                self.result_title_var.set("")
                self.result_content_text.delete(1.0, tk.END)
                self.prev_button.config(state=tk.DISABLED)
                self.next_button.config(state=tk.DISABLED)
            else:
                self._update_search_display()
                self.search_status_var.set(
                    f"Displaying result {self.current_index + 1} of {len(self.current_results)}"
                )
                self.prev_button.config(
                    state=tk.NORMAL if len(self.current_results) > 1 else tk.DISABLED
                )
                self.next_button.config(
                    state=tk.NORMAL if len(self.current_results) > 1 else tk.DISABLED
                )

            self.search_active = True
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to search notes: {str(e)}")

    def _update_search_display(self) -> None:
        """Update the search result display"""
        if 0 <= self.current_index < len(self.current_results):
            note_id, title, content = self.current_results[self.current_index]
            self.result_title_var.set(title)
            self.result_content_text.delete(1.0, tk.END)
            self.result_content_text.insert(tk.END, content)
            self.search_status_var.set(
                f"Displaying result {self.current_index + 1} of {len(self.current_results)}"
            )

    def _show_previous_result(self) -> None:
        """Show the previous search result"""
        if self.current_index > 0:
            self.current_index -= 1
            self._update_search_display()
            self.next_button.config(state=tk.NORMAL)

            if self.current_index == 0:
                self.prev_button.config(state=tk.DISABLED)

    def _show_next_result(self) -> None:
        """Show the next search result"""
        if self.current_index < len(self.current_results) - 1:
            self.current_index += 1
            self._update_search_display()
            self.prev_button.config(state=tk.NORMAL)

            if self.current_index == len(self.current_results) - 1:
                self.next_button.config(state=tk.DISABLED)

    def run(self) -> None:
        """Run the application"""
        self.root.mainloop()

    def __del__(self) -> None:
        """Close the database connection when the object is destroyed"""
        if self.connection:
            self.connection.close()


if __name__ == "__main__":
    root = tk.Tk()
    app = NotepadApp(root)
    app.run()
