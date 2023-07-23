from speakListen import hear
from speakListen import speak
import docx
import fitz
import time
from rich.console import Console # pip3 install Rich
from rich.table import Table
from colorama import Fore

def ms_word():
    """[Print and speak out a ms_word docx file as specified in the path]
    """    
    # TODO : Take location input from the user
    try:
        speak("Enter the document's location - ")
        location = input("Enter the document's location - ")
        
        file_loc = doubleslash(location) 
       
        doc = docx.Document(file_loc)
        fullText = []
        for para in doc.paragraphs:
            fullText.append(para.text)
        #print(fullText)
        doc_file = '\n'.join(fullText)
        print(doc_file)
        speak(doc_file)
    except Exception as exp:
        #print(exp)
        print(f"ERROR - {exp}")
        print(Fore.YELLOW + "I could'nt locate the file!\nIf you didn't specify the extension of the file, please specify it.")
        return "None"

def pdf_read():
    """[Print and speak out the pdf on specified path]
    """    
    try:
        speak("Enter the document's location - ")
        location = input("Enter the document's location - ")
        
        path = doubleslash(location) 
        pdf = fitz.open(path)
        details = pdf.metadata # Stores the meta-data which generally includes Author name and Title of book/document.
        total_pages = pdf.pageCount # Stores the total number of pages

    except Exception as exp:
        print(f"ERROR - {exp}")
        print(Fore.YELLOW + "I could'nt locate the file!\nIf you didn't specify the extension of the file, please specify it.")
        return "None"
    try :
        """     1. Author
                2. Creator
                3. Producer
                4. Title  """   
        
        author =  details["author"]
        #print("Author : ",author)
        
        title = details["title"]
        #print("Title : ",title) 
            
        #print(details)
        #print("Total Pages : ",total_pages)
        book_details(author, title, total_pages)
        speak(f" Title {title}")
        speak(f" Author {author}")
        speak(f" Total Pages {total_pages}")
        
        # TODO : Deal with the Index
        toc = pdf.get_toc()
        print("Say 1 or \"ONLY PRINT INDEX\" - if you want me to print the book's index.\nSay 2 if you want me to print and make me speak out the book's index.\nSay any key if you don't want to print the index.'")
        speak("Say 1 or only print index if you want me to print the book's index.\nSay 2 if you want me to print and make me speak out the book's index.\nSay any key if you don't want to print the index.'")
        q = hear().lower()

        if "only print" in q or "1" in q or "one" in q or "vone" in q or 'only' in q or "index only" in q or 'only' in q or "print only" in q: 
            print_index(toc)
            time.sleep(15)
        elif "speak" in q or "2" in q or 'two' in q: 
            print_n_speak_index(toc)
            time.sleep(10)
        elif q == "None":
            print("I could'nt understand what you just said!")
            speak("I could'nt understand what you just said!")
            time.sleep(4)
        else: 
            time.sleep(4)
            pass

    
        """Allow the user to do the following
        1. Read/speak a page
        2. Read/speak a range of pages
        3. Lesson
        4. Read/speak a whole book
        """  
        
        #time.sleep(5)
  
        print("____________________________________________________________________________________________________________")
        print("1. Print/speak a single page\n2. Print/speak a range of pages\n3. Print/speak a Lesson\n4. Read/speak a whole book")
        speak("1. Print/speak a single page\n2. Print/speak a range of pages\n3. Print/speak a Lesson\n4. Read/speak a whole book")
        q = hear().lower()
        if "single" in q or "one" in q or "vone" in q or "one page" in q or "vone page" in q or "1 page" in q:
            try:
                pgno = int(input("Page Number - "))

                page = pdf.load_page(pgno - 1)
                text = page.get_text('text')
                print("\n\n")
                print(text.replace('\t',' '))
                speak(text.replace('\t',' '))
            except Exception:
                print("Sorry, I could recognize what you entered. Please re-enter the Page Number.")
                speak("Sorry, I could recognize what you entered. Please re-enter the Page Number.")
                pgno = input("Page no. - ")
                page = pdf.load_page(pgno - 1)
                text = page.get_text('text')
                print(text.replace('\t',' '))
                speak(text.replace('\t',' '))

        
        elif 'range' in q or "multiple" in q:
            try:
                start_pg_no = int(input("Starting Page Number - "))
                end_pg_no = int(input("End Page Number - "))
                for i in range(start_pg_no - 1, end_pg_no):
                    page = pdf.load_page(i)
                    text = page.get_text('text')
                    print(text.replace('\t',' '))
                    speak(text.replace('\t',' '))
            except Exception:
                print("Sorry, I could recognize what you entered. Please re-enter the Page Number.")
                speak("Sorry, I could recognize what you entered. Please re-enter the Page Number.")
                start_pg_no = int(input("Starting Page Number - "))
                end_pg_no = int(input("End Page Number - "))
                for i in range(start_pg_no - 1, end_pg_no - 1):
                    page = pdf.load_page(i)
                    text = page.get_text('text')
                    print(text.replace('\t',' '))
                    speak(text.replace('\t',' '))

        elif 'lesson' in q:
            try:
                key = input("Lesson name - ")
                start_pg_no, end_pg_no = search_in_toc(toc, key, total_pages)
                if start_pg_no != None and end_pg_no != None:
                    start_pg_no, end_pg_no = map(int,search_in_toc(toc, key, total_pages))
                
                    for i in range(start_pg_no - 1, end_pg_no):
                        page = pdf.load_page(i)
                        text = page.get_text('text')
                        print(text.replace('\t',' '))
                        speak(text.replace('\t',' '))
                else: 
                    print("Try Again.")
                    speak("Try Again.")
                    speak("Lesson name")
                    key = input("Lesson name - ")
                    start_pg_no, end_pg_no = map(int,search_in_toc(toc, key, total_pages))
                    if start_pg_no != None and end_pg_no != None:
                        for i in range(start_pg_no - 1, end_pg_no):
                            page = pdf.load_page(i)
                            text = page.get_text('text')
                            print(text.replace('\t',' '))
                            speak(text.replace('\t',' '))
                    
            except Exception:
                print("Try Again! Lesson could not be found.")
                speak("Try Again.Lesson could not be found")
                speak("Lesson name")
                key = input("Lesson name - ")
                start_pg_no, end_pg_no = search_in_toc(toc, key, total_pages)
                if start_pg_no != None and end_pg_no != None:
                    start_pg_no, end_pg_no = map(int,search_in_toc(toc, key, total_pages))
                
                    for i in range(start_pg_no - 1, end_pg_no):
                        page = pdf.load_page(i)
                        text = page.get_text('text')
                        print(text.replace('\t',' '))
                        speak(text.replace('\t',' '))
                else: 
                    print("Sorry, I cannot find the perticular lesson.")
                    speak("Sorry, I cannot find the perticular lesson.")

        elif "whole" in q or 'complete' in q:
            for i in range(total_pages):
                page = pdf.load_page(i)
                text = page.get_text('text')
                print(text.replace('\t',' '))
                speak(text.replace('\t',' '))

        elif q == "None":
            print("I could'nt understand what you just said!")
            speak("I could'nt understand what you just said!")
        else:
            print("You didn't say a valid command!")
            time.sleep(5)
    except Exception as e: 
        print(e)
    pass
    pdf.close()

def doubleslash(text):
    """Replaces / with // 

    Args:
        text (str): location

    Returns:
        str: formatted location
    """
    return text.replace('\\' , '\\\\')

def print_index(toc):
    """Prints out the index in proper format with title name and page number

    Args:
        toc (nested list): toc[1] - Topic name
                           toc[2] - Page number
    """
    dash = "-"*(100 - 7)
    space = " "*47
    print(f"{space}INDEX")
    print(f"\n\nName : {dash} PageNo.\n\n\n")
    for topic in toc:
        eq_dash = "-"*(100 - len(topic[1]))
        print(f"{topic[1]} {eq_dash} {topic[2]}")
        
def print_n_speak_index(toc):
    """Along with printing, it speaks out the index too.

    Args:
        toc (nested list): toc[1] - Topic name
                           toc[2] - Page number
    """
    dash = "-"*(100 - 7)
    space = " "*47
    print(f"{space}INDEX")
    print(f"\n\nName : {dash} PageNo.\n\n\n\n")
    for topic in toc:
        eq_dash = "-"*(100 - len(topic[1]))
        print(f"{topic[1]} {eq_dash} {topic[2]}")
        speak(f"{topic[1]}  {topic[2]}")

def search_in_toc(toc, key, totalpg):
    """Searches a particular lesson name provided as a parameter in toc and returns its starting and ending page numbers.

    Args:
        toc (nested list): toc[1] - Topic name
                           toc[2] - Page number
        key (str): the key to be found
        totalpg (int): total pages in book/document

    Returns:
        int: staring and ending page numbers of lesson found.
        If not found then return None
    """
    for i in range(len(toc) - 1):
        topic = toc[i]
        if i != len(toc) - 2:
            if topic[1] == key:
                nexttopic = toc[i + 1]
                return (topic[2], nexttopic[2])
            elif topic[1].lower() == key:
                nexttopic = toc[i + 1]
                return (topic[2], nexttopic[2])
        else:
            if topic[1] == key:
                return (topic[2], totalpg)
            elif topic[1].lower() == key:
               
                return (topic[2], totalpg)
    return None,None

def book_details(author, title, total_pages):
    """Creates a table of book details like author name, title, and total pages.

    Args:
        author (str): Name of author
        title (str): title of the book
        total_pages (int): total pages in the book
    """
    table = Table(title="\nBook Details :- ", show_lines = True) 

    table.add_column("Sr. No.", style="magenta", no_wrap=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", justify="left", style="green")

    table.add_row("1", "Title", f"{title}")
    table.add_row("2", "Author", f"{author}")
    table.add_row("3", "Pages", f"{total_pages}")

    console = Console()
    console.print(table)
   
#ms_word()
#pdf_read()
#book_details("abc", "abcde", 12)
