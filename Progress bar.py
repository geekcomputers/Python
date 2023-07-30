import math
import colorama
from colorama import Fore, Style

colorama.init()

def progress_bar(progress, total, color=Fore.CYAN, complete_color=Fore.GREEN):
    """
    - progress:int   --> current progress of the operation (ex. index of an item inside a list)
    - total:int      --> the actual lenght of the operation (ex. the lenght of a list of items)
    - color          --> color for the bar and percentage value during the operation
    - complete_color --> color for the bar and percentage value once the operation is complete
    """

    percent = 100 * (progress / total)
    bar_length = 50 #Set fixed lenght of the bar to 50 chars (Thanks to NitkarshChourasia for improvement)
    completed_length = int(bar_length * (progress / total))
    bar = '█' * completed_length + '-' * (bar_length - completed_length)

    print(f'\r|{color}{bar}{Style.RESET_ALL}| {color}{percent:.2f}%{Style.RESET_ALL}', end='', flush=True)
    #Using f-strings and print statement's parameter:
    #\r --> Special escape character that allows to keep writing on the same section of the line (in this case from the beginning)
    #{color + bar + Style.RESET_ALL} --> Sums the Fore color character to create a pretty colored bar on the screen
    #percent:.2f --> ":.xf" allows to round to x value of decimal points (2 in this case). It is important, for this script, that the two print statement use the same x value to avoid misprinting
    #{color}{percent:.2f}{Style.RESET_ALL} --> Same as previous but since "percent" is of type float it can't be directly added to a string. 
    #                                          To fix this you can either use this sintax or convert it to str by doing {color + str(percent:.2f) + Style.RESET_ALL}
    #flush=True --> ensure real-time printing of the progress bar without buffering

    if progress == total:
        print(f'\r|{complete_color}{bar}{Style.RESET_ALL}| {complete_color}{percent:.2f}%{Style.RESET_ALL}')

numbers = [x * 5 for x in range(2000, 3000)]
lenght = len(numbers)

progress_bar(0, total=lenght)

results = []
for i, x in enumerate(numbers):
    #Iterate over the list of ints with enumerate to get both the index "i" (used for progress) and the value of the int "x" to execute the operation
    results.append(math.factorial(x))
    progress_bar(progress=(i + 1), total=lenght)
    #Since indexes start at 0, add 1 to the first iteration and increase it by one at every following iteration for consistency

print("\nFactorials calculated successfully!")
