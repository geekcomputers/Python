"""
Tkinter Clock Difference Calculator.

Compute difference between two times (HH:MM:SS) with midnight wrap-around.

Doctests:

>>> clock_diff("12:00:00", "14:30:15")
'02:30:15'
>>> clock_diff("23:50:00", "00:15:30")
'00:25:30'
>>> clock_diff("00:00:00", "00:00:00")
'00:00:00'
"""

import tkinter as tk
from tkinter import messagebox


def clock_diff(t1: str, t2: str) -> str:
    """Return difference between t1 and t2 as HH:MM:SS (zero-padded)."""
    h1, m1, s1 = int(t1[0:2]), int(t1[3:5]), int(t1[6:8])
    h2, m2, s2 = int(t2[0:2]), int(t2[3:5]), int(t2[6:8])
    sec1 = h1 * 3600 + m1 * 60 + s1
    sec2 = h2 * 3600 + m2 * 60 + s2
    diff = sec2 - sec1
    if diff < 0:
        diff += 24 * 3600
    h = diff // 3600
    m = (diff % 3600) // 60
    s = diff % 60
    return f"{h:02}:{m:02}:{s:02}"


def calculate() -> None:
    """Tkinter callback to calculate and display clock difference."""
    t1 = entry_t1.get().strip()
    t2 = entry_t2.get().strip()
    try:
        for t in [t1, t2]:
            if len(t) != 8 or t[2] != ":" or t[5] != ":":
                raise ValueError("Format must be HH:MM:SS")
            h, m, s = int(t[0:2]), int(t[3:5]), int(t[6:8])
            if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
                raise ValueError("Time out of range")
        result = clock_diff(t1, t2)
        label_result.config(text=f"Difference: {result}")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input!\n{e}")


root = tk.Tk()
root.title("Clock Difference Calculator")
root.geometry("300x200")

tk.Label(root, text="Init schedule (HH:MM:SS):").pack(pady=5)
entry_t1 = tk.Entry(root)
entry_t1.pack()

tk.Label(root, text="Final schedule (HH:MM:SS):").pack(pady=5)
entry_t2 = tk.Entry(root)
entry_t2.pack()

tk.Button(root, text="Calculate Difference", command=calculate).pack(pady=10)
label_result = tk.Label(root, text="Difference: ")
label_result.pack(pady=5)

if __name__ == "__main__":
    import doctest

    doctest.testmod()
    root.mainloop()
