import math, colorama
from colorama import Fore

colorama.init()
def progress_bar(progress:int = 0, total:int = 100, color=Fore.CYAN, complete_color=Fore.GREEN):
    """
    - progress:int   --> current progress of the operation (ex. index of an item inside a list)
    - total:int      --> the actual lenght of the operation (ex. the lenght of a list of items)
    - color          --> color for the bar and percentage value during the operation
    - complete_color --> color for the bar and percentage value once the operation is complete
    """
    
    percent = 100 * (progress / float (total)) #Calculate the percentage of completion by multiplying the ratio of progress over total for 100
    bar = "█" * int(percent) + "-" * (100 - int(percent)) #Create the actual bar by multiplying the bar character for the percentage of completion and multiplying the remaining character for the difference 

    print(f"\r|{color + bar + Fore.RESET}| {color}{percent:.2f}{Fore.RESET}%", end="\r")
    #Using f-strings and print statement's parameter:
    #\r --> Special escape character that allows to keep writing on the same section of the line (in this case from the beginning)
    #{color + bar + Fore.RESET} --> Sums the Fore color character to create a pretty colored bar on the screen
    #percent:.2f --> The :.xf allows to round to x value of decimal points (2 in this case). It is important, for this script, that the two print statement use the same x value to avoid misprinting
    #{color}{percent:.2f}{Fore.RESET} --> Same as previous but since "percent" is a float it can't be directly added to a string. 
    #                                     To fix this you can either use this sintax or convert it yourself to str by doing {color + str(percent:.2f) + Fore.RESET}
    #end="\r" --> The "end" parameter specify how the line print should end. Here "\r" goes back to the previously printed one (at the start of the line)

    if progress == total: 
        print(f"\r|{complete_color + bar + Fore.RESET}| {complete_color}{percent:.2f}{Fore.RESET}%", end="\n")
        #Same as the previous print statement but here it changes the color of the complete bar and add a new line indicaton at the end of the line (end="\n")

#The simplest imlpementation of this progressbar script I could think of
numbers = [x * 5 for x in range (2000, 3000)] #Generate a list of 1000 ints
results = []
lenght = len(numbers) #Get the total lenght of the operation

progress_bar (0, total=lenght, color=Fore.CYAN, complete_color=Fore.GREEN)
# Set the bar progress to 0

for i, x in enumerate(numbers): 
    #Iterate over the list of ints with enumerate to get both the index "i" (used for progress) and the value of the int "x" to execute the operation
    results.append(math.factorial(x))
    progress_bar(i + 1, total=lenght, color=Fore.CYAN, complete_color=Fore.GREEN)
    #Expaination:
    #i + 1 --> Since indexes start at "0", we add 1 to give the progress value to the first iteration and increase it by one at every following iteration for consistency
    #total=lenght --> Used to calculate the percentage of progress. In this case is the lenght of the list of numbers to factorialize
    #color=Fore.CYAN --> Is the color of the progress bar during the operation
    #complete_color=Fore.GREEN --> Is the color of the bar on completion
