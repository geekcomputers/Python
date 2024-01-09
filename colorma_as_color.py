import colorama as color


from colorama import Fore, Back, Style

print(Fore.RED + "some red text")
print(Back.GREEN + "and with a green background")
print("So any text will be in green background?")

print("So is it a wrapper of some sort?")
print("dark_angel wasn't using it in her code.")
print("she was just being using direct ANSI codes.")
print(Style.RESET_ALL)
print(Fore.BRIGHT_RED + "some bright red text")
print(Back.WHITE + "and with a white background")
print("Will need to study about what is ANSI codes.")
print(Style.DIM + "and in dim text")
print(Style.RESET_ALL)
print("back to normal now")


# …or, Colorama can be used in conjunction with existing ANSI libraries such as the venerable Termcolor the fabulous Blessings, or the incredible _Rich.