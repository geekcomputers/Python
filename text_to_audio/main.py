# A exclusive CLI version can be made using inquirer library.
from gtts import gTTS
from io import BytesIO

# only use when needed to avoid memory usage in program
from pprint import pprint

"""_summary_
def some_function():
    # Pygame module is only imported when this function is called
    import pygame.mixer as mixer
    mixer.init()

# USE LAZY LOADING

    Returns:
        _type_: _description_
    """

"""
# For example, if you are using pygame, you might do something like:
# import pygame
# audio_file.seek(0)  # Reset the BytesIO object to the beginning
# pygame.mixer.init()
# pygame.mixer.music.load(audio_file)
# pygame.mixer.music.play()

# Note: The actual loading and playing of the MP3 data in an audio library are not provided in the code snippet.
# The last comments indicate that it depends on the specific audio library you choose.

"""
# Should have

# How to play a audio without saving it?
# efficiently?
# So I can also combine two languages?
# Exception for network issues?

# class userAudio:

# print("\n")
# print(dir(gTTS))

# file_naming can be added too.


class userAudio:
    def __init__(
        self,
        text: str = None,
        language: str = "en",
        slow: bool = True,
        accent: str = "com",
    ):  # Correct the syntax here.
        self.lang = language
        self.slow = slow
        self.accent = accent

        if text is None:
            self.user_input()
        else:
            self.text_to_audio = text

        self.gtts_object = gTTS(
            text=self.text_to_audio, lang=self.lang, slow=self.slow, tld=self.accent
        )

    # ! Some error is here.
    def user_input(self):
        text = input("Enter the text you want to convert to audio: ")
        self.text_to_audio = text
        self.gtts_object = gTTS(
            text=self.text_to_audio, lang=self.lang, slow=self.slow, tld=self.accent
        )  # Just need to understand the class workings little better.
        # Isn't this declaring this again?

    def save_only(self, filename="default.mp3"):
        # The class will take care of the playing and saving.
        # The initialisation will take care of it.
        self.gtts_object.save(filename)

    def play_only(self):
        from pygame import mixer, time

        tts = self.gtts_object
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)  # Reset the BytesIO object to the beginning
        mixer.init()
        mixer.music.load(fp)
        mixer.music.play()
        while mixer.music.get_busy():
            time.Clock().tick(10)
        # Consider using a different method for playing audio, Pygame might not be optimal

    # Object initialisation please.
    # def save_path(self):
    #     from pathlib import Path

    #     user_path = Path(input("Enter the path to save the audio: "))

    #     # .exists() is a method in Path class
    #     if user_path.exists:
    #         pprint(f"The provided path {user_path} exists.")
    #         # full_path = user_path + "/" + input("Enter the file name: ")
    #         full_path = user_path + "/" + "default.mp3"
    #         self.save(user_path)
    #         pprint("File saved successfully")
    #     else:
    #         # prompts the user again three times to do so.
    #         # if not then choose the default one asking user to choose the default one.
    #         # if he says no, then asks to input again.
    #         # then ask three times.
    #         # at max
    #     """dir testing has to be done seprately"""

    #     if user_path.is_dir:
    #         gTTS.save(user_path)

    # def file_name(self):
    #     while True:
    #         file_path = input("Enter the file path: ")
    #         if file_path.exists:
    #             break
    #         else:
    #             # for wrong input type exceptions
    #             while True:
    #                 continue_response = input("Are you sure you want to continue?(y/n):")
    #                 continue_response = continue_response.strip().lower()
    #                 if continue_response in ["y", "yes", "start"]:
    #                     break
    #                 elif continue_response in ["n", "no", "stop"]:
    #                     break
    #     # file_path = user_path + "/" + input("Enter the file name: ")
    #     # file_path = user_path + "/" + "default.mp3"
    #     # Also work a best way to save good quality audio and what is best format to save it in.

    # def save_and_play(self):
    #     self.save_only()
    #     self.play_only()
    #     self.save_path()
    #     self.file_name()

    # def concatenate_audio(self):
    #     # logic to concatenate audio?
    #     # why, special feature about it?
    #     # this is not a logic concatenation application.
    #     pass


# hello = userAudio("Hello, world!")
# hello.play_only()

with open("special_file.txt", "r") as f:
    retrieved_text = f.read()
retrieved_text = retrieved_text.replace("\n", "")

# hello = userAudio("Hello, user how are you?", slow=False)
hello = userAudio
hello.play_only()


class fun_secret_generator_string:
    # Instructions on how to use it?
    def __init__(self, string):
        self.string = string

    # text = "Input your text here."
    # with open("special_file.txt", "w") as f:
    #     for char in text:
    #         f.write(char + "\n")
    #     f.close()
    #     print("File saved successfully")

    # Reading from the file
    with open("special_file.txt", "r") as f:
        retrieved_text = f.read()
    retrieved_text = retrieved_text.replace("\n", "")


# Also have an option to play from a file, a text file.
# Will later put other pdf and word2docx vectorisations.
# from gtts import gTTS
# import os

# # Enter the name of your text file
# mytextfile = "hello.txt"

# # Specify the language in which you want your audio
# language = "en"

# # Get the contents of your file
# with open(mytextfile, 'r') as f:
#     mytext = f.read()
#     f.close()

# # Create an instance of gTTS class
# myobj = gTTS(text=mytext, lang=language, slow=False)

# # Method to create your audio file in mp3 format
# myobj.save("hello.mp3")
# print("Audio Saved")

# # This will play your audio file
# os.system("mpg321 hello.mp3")
