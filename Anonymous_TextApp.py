import tkinter as tk
from PIL import Image, ImageTk
from twilio.rest import Client

window = tk.Tk()
window.title("Anonymous_Text_App")
window.geometry("800x750")

# Define global variables
body = ""
to = ""


def message():
    global body, to
    account_sid = "Your_account_sid"  # Your account sid
    auth_token = "Your_auth_token"  # Your auth token
    client = Client(account_sid, auth_token)
    msg = client.messages.create(
        from_="Twilio_number",  # Twilio number
        body=body,
        to=to,
    )
    print(msg.sid)
    confirmation_label.config(text="Message Sent!")


try:
    # Load the background image
    bg_img = Image.open(r"D:\Downloads\img2.png")

    # Canvas widget
    canvas = tk.Canvas(window, width=800, height=750)
    canvas.pack(fill="both", expand=True)

    #  background image to the Canvas
    bg_photo = ImageTk.PhotoImage(bg_img)
    bg_image_id = canvas.create_image(0, 0, image=bg_photo, anchor="nw")
    bg_image_id = canvas.create_image(550, 250, image=bg_photo, anchor="center")
    bg_image_id = canvas.create_image(1100, 250, image=bg_photo, anchor="center")
    bg_image_id = canvas.create_image(1250, 250, image=bg_photo, anchor="center")
    bg_image_id = canvas.create_image(250, 750, image=bg_photo, anchor="center")
    bg_image_id = canvas.create_image(850, 750, image=bg_photo, anchor="center")
    bg_image_id = canvas.create_image(1300, 750, image=bg_photo, anchor="center")

    # Foreground Image
    img = Image.open(r"D:\Downloads\output-onlinepngtools.png")
    photo = ImageTk.PhotoImage(img)
    img_label = tk.Label(window, image=photo, anchor="w")
    img_label.image = photo
    img_label.place(x=10, y=20)

    # Text for number input
    canvas.create_text(
        1050,
        300,
        text="Enter the number starting with +[country code]",
        font=("Poppins", 18, "bold"),
        fill="black",
        anchor="n",
    )
    text_field_number = tk.Entry(
        canvas,
        width=17,
        font=("Poppins", 25, "bold"),
        bg="#404040",
        fg="white",
        show="*",
    )
    canvas.create_window(1100, 350, window=text_field_number, anchor="n")

    # Text for message input
    canvas.create_text(
        1050,
        450,
        text="Enter the Message",
        font=("Poppins", 18, "bold"),
        fill="black",
        anchor="n",
    )
    text_field_text = tk.Entry(
        canvas, width=17, font=("Poppins", 25, "bold"), bg="#404040", fg="white"
    )
    canvas.create_window(1100, 500, window=text_field_text, anchor="n")

    #  label for confirmation message
    confirmation_label = tk.Label(window, text="", font=("Poppins", 16), fg="green")
    canvas.create_window(1100, 600, window=confirmation_label, anchor="n")

except Exception as e:
    print(f"Error loading image: {e}")


# Function to save input and send message
def save_and_send():
    global body, to
    to = str(text_field_number.get())
    body = str(text_field_text.get())
    message()


# Button to save input and send message
save_button = tk.Button(window, text="Save and Send", command=save_and_send)
canvas.create_window(1200, 550, window=save_button, anchor="n")

window.mainloop()
