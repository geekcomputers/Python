from tkinter import *

# ---------------------------- CONSTANTS & GLOBALS ------------------------------- #
PINK = "#e2979c"
GREEN = "#9bdeac"
FONT_NAME = "Courier"
DEFAULT_WORK_MIN = 25
DEFAULT_BREAK_MIN = 5

# Background color options
bg_colors = {
    "Pink": "#e2979c",
    "Green": "#9bdeac",
    "Blue": "#1f75fe",
    "Yellow": "#ffcc00",
    "Purple": "#b19cd9",
}

# Global variables
ROUND = 1
timer_mec = None
total_time = 0  # Total seconds for the current session
is_paused = False  # Timer pause flag
remaining_time = 0  # Remaining time (in seconds) when paused
custom_work_min = DEFAULT_WORK_MIN
custom_break_min = DEFAULT_BREAK_MIN


# ---------------------------- BACKGROUND COLOR CHANGE FUNCTION ------------------------------- #
def change_background(*args):
    selected = bg_color_var.get()
    new_color = bg_colors.get(selected, PINK)
    window.config(bg=new_color)
    canvas.config(bg=new_color)
    label.config(bg=new_color)
    tick_label.config(bg=new_color)
    work_label.config(bg=new_color)
    break_label.config(bg=new_color)


# ---------------------------- NOTIFICATION FUNCTION ------------------------------- #
def show_notification(message):
    notif = Toplevel(window)
    notif.overrideredirect(True)
    notif.config(bg=PINK)

    msg_label = Label(
        notif,
        text=message,
        font=(FONT_NAME, 12, "bold"),
        bg=GREEN,
        fg="white",
        padx=10,
        pady=5,
    )
    msg_label.pack()

    window.update_idletasks()
    wx = window.winfo_rootx()
    wy = window.winfo_rooty()
    wwidth = window.winfo_width()
    wheight = window.winfo_height()

    notif.update_idletasks()
    nwidth = notif.winfo_width()
    nheight = notif.winfo_height()

    x = wx + (wwidth - nwidth) // 2
    y = wy + wheight - nheight - 10
    notif.geometry(f"+{x}+{y}")

    notif.after(3000, notif.destroy)


# ---------------------------- TIMER FUNCTIONS ------------------------------- #
def reset_timer():
    global ROUND, timer_mec, total_time, is_paused, remaining_time
    ROUND = 1
    is_paused = False
    remaining_time = 0
    if timer_mec is not None:
        window.after_cancel(timer_mec)
    canvas.itemconfig(timer_text, text="00:00")
    label.config(text="Timer")
    tick_label.config(text="")
    total_time = 0
    canvas.itemconfig(progress_arc, extent=0)
    start_button.config(state=NORMAL)
    pause_button.config(state=DISABLED)
    play_button.config(state=DISABLED)


def start_timer():
    global ROUND, total_time, is_paused
    canvas.itemconfig(progress_arc, extent=0)

    if ROUND % 2 == 1:  # Work session
        total_time = custom_work_min * 60
        label.config(text="Work", fg=GREEN)
    else:  # Break session
        total_time = custom_break_min * 60
        label.config(text="Break", fg=PINK)

    count_down(total_time)
    start_button.config(state=DISABLED)
    pause_button.config(state=NORMAL)
    play_button.config(state=DISABLED)
    is_paused = False


def count_down(count):
    global timer_mec, remaining_time
    remaining_time = count
    minutes = count // 60
    seconds = count % 60
    if seconds < 10:
        seconds = f"0{seconds}"
    canvas.itemconfig(timer_text, text=f"{minutes}:{seconds}")

    if total_time > 0:
        progress = (total_time - count) / total_time
        canvas.itemconfig(progress_arc, extent=progress * 360)

    if count > 0 and not is_paused:
        timer_mec = window.after(1000, count_down, count - 1)
    elif count == 0:
        if ROUND % 2 == 1:
            show_notification("Work session complete! Time for a break.")
        else:
            show_notification("Break over! Back to work.")
        if ROUND % 2 == 0:
            tick_label.config(text=tick_label.cget("text") + "#")
        ROUND += 1
        start_timer()


def pause_timer():
    global is_paused, timer_mec
    if not is_paused:
        is_paused = True
        if timer_mec is not None:
            window.after_cancel(timer_mec)
        pause_button.config(state=DISABLED)
        play_button.config(state=NORMAL)


def resume_timer():
    global is_paused
    if is_paused:
        is_paused = False
        count_down(remaining_time)
        play_button.config(state=DISABLED)
        pause_button.config(state=NORMAL)


def set_custom_durations():
    global custom_work_min, custom_break_min
    try:
        work_val = int(entry_work.get())
        break_val = int(entry_break.get())
        custom_work_min = work_val
        custom_break_min = break_val
        canvas.itemconfig(left_custom, text=f"{custom_work_min}m")
        canvas.itemconfig(right_custom, text=f"{custom_break_min}m")
    except ValueError:
        pass


# ---------------------------- UI SETUP ------------------------------- #
window = Tk()
window.title("Pomodoro")
window.config(padx=100, pady=50, bg=PINK)

# Canvas setup with increased width for spacing
canvas = Canvas(window, width=240, height=224, bg=PINK, highlightthickness=0)
timer_text = canvas.create_text(
    120, 112, text="00:00", font=(FONT_NAME, 35, "bold"), fill="white"
)
background_circle = canvas.create_arc(
    40, 32, 200, 192, start=0, extent=359.9, style="arc", outline="white", width=5
)
progress_arc = canvas.create_arc(
    40, 32, 200, 192, start=270, extent=0, style="arc", outline="green", width=5
)
# Updated positions for work and break time labels
left_custom = canvas.create_text(
    20, 112, text=f"{custom_work_min}m", font=(FONT_NAME, 12, "bold"), fill="white"
)
right_custom = canvas.create_text(
    220, 112, text=f"{custom_break_min}m", font=(FONT_NAME, 12, "bold"), fill="white"
)

canvas.grid(column=1, row=1)

label = Label(text="Timer", font=(FONT_NAME, 35, "bold"), bg=PINK, fg="green")
label.grid(column=1, row=0)

start_button = Button(text="Start", command=start_timer, highlightthickness=0)
start_button.grid(column=0, row=2)

reset_button = Button(text="Reset", command=reset_timer, highlightthickness=0)
reset_button.grid(column=2, row=2)

pause_button = Button(
    text="Pause", command=pause_timer, highlightthickness=0, state=DISABLED
)
pause_button.grid(column=0, row=3)

play_button = Button(
    text="Play", command=resume_timer, highlightthickness=0, state=DISABLED
)
play_button.grid(column=2, row=3)

tick_label = Label(text="", font=(FONT_NAME, 15, "bold"), bg=PINK, fg="green")
tick_label.grid(column=1, row=4)

# Custom durations (stacked vertically)
work_label = Label(
    text="Work (min):", font=(FONT_NAME, 12, "bold"), bg=PINK, fg="white"
)
work_label.grid(column=1, row=5, pady=(20, 0))
entry_work = Entry(width=5, font=(FONT_NAME, 12))
entry_work.grid(column=1, row=6, pady=(5, 10))
break_label = Label(
    text="Break (min):", font=(FONT_NAME, 12, "bold"), bg=PINK, fg="white"
)
break_label.grid(column=1, row=7, pady=(5, 0))
entry_break = Entry(width=5, font=(FONT_NAME, 12))
entry_break.grid(column=1, row=8, pady=(5, 10))
set_button = Button(
    text="Set Durations", command=set_custom_durations, font=(FONT_NAME, 12)
)
set_button.grid(column=1, row=9, pady=(10, 20))

# OptionMenu for changing background color
bg_color_var = StringVar(window)
bg_color_var.set("Pink")
bg_option = OptionMenu(
    window, bg_color_var, *bg_colors.keys(), command=change_background
)
bg_option.config(font=(FONT_NAME, 12))
bg_option.grid(column=1, row=10, pady=(10, 20))

window.mainloop()
