"""@Author: Anurag Kumar(mailto:anuragkumarak95@gmail.com) 
This module is used for generating a TF-IDF file or values from a list of files that contains docs.

What is TF-IDF : https://en.wikipedia.org/wiki/Tf%E2%80%93idf

python:
  - 3.5

pre-requisites: 
  - colorama==0.3.9 

sample file format of input:

    ##START(NOT INCLUDED)
    sport smile today because signs Gemini
    little sister dealt severe allergy figure
    about looks gender color attitude nationality respect
    added video playlist Sonic Fightstick Edition
    weeks birthday scott wants camping keeper
    photo taking photo trying auction scale photo
    happy creatively capture story stage magical
    yoongi looks seokjin looking yoongi looking seokjin
    taking glasses because buffering cannot handle
    tried Michelle Obama proceeded defend whole pointless
    robbed shades backstage reading guess karma stealing
    remains sailors destroyer McCain collision found
    timeline beginnings infographics Catch upcoming debut
    ##END(NOT INCLUDED)

here, every line represents a document.

have fun, cheers.
"""
import math
import pickle

from colorama import Fore, Style

switcher = {
    "r": Fore.RED,
    "bk": Fore.BLACK,
    "b": Fore.BLUE,
    "g": Fore.GREEN,
    "y": Fore.YELLOW,
    "m": Fore.MAGENTA,
    "c": Fore.CYAN,
    "w": Fore.WHITE,
}


def paint(str, color="r"):
    """Utility func, for printing colorful logs in console...

    @args:
    --
    str : String to be modified.
    color : color code to which the string will be formed. default is 'r'=RED

    @returns:
    --
    str : final modified string with foreground color as per parameters.

    """
    if color in switcher:
        str = switcher[color] + str + Style.RESET_ALL
    return str


TAG = paint("TF-IDF-GENE/", "b")


def find_tf_idf(file_names=None, prev_file_path=None, dump_path=None):
    """Function to create a TF-IDF list of dictionaries for a corpus of docs.
    If you opt for dumping the data, you can provide a file_path with .tfidfpkl extension(standard made for better understanding)
    and also re-generate a new tfidf list which overrides over an old one by mentioning its path.

    @Args:
    --
    file_names : paths of files to be processed on, you can give many small sized file, rather than one large file.
    prev_file_path : path of old .tfidfpkl file, if available. (default=None)
    dump_path : directory-path where to dump generated lists.(default=None)

    @returns:
    --
    idf : a dict of unique words in corpus,with their document frequency as values.
    tf_idf : the generated tf-idf list of dictionaries for mentioned docs.
    """
    if file_names is None:
        file_names = ["./../test/testdata"]
    tf_idf = (
        []
    )  # will hold a dict of word_count for every doc(line in a doc in this case)
    idf = {}

    # this statement is useful for altering existant tf-idf file and adding new docs in itself.(## memory is now the biggest issue)
    if prev_file_path:
        print(TAG, "modifying over exising file.. @", prev_file_path)
        idf, tf_idf = pickle.load(open(prev_file_path, "rb"))
        prev_doc_count = len(idf)
        prev_corpus_length = len(tf_idf)

    for f in file_names:

        file1 = open(
            f, "r"
        )  # never use 'rb' for textual data, it creates something like,  {b'line-inside-the-doc'}

        # create word_count dict for all docs
        for line in file1:
            dict = {}
            # find the amount of doc a word is in
            for i in set(line.split()):
                if i in idf:
                    idf[i] += 1
                else:
                    idf[i] = 1
            for word in line.split():
                # find the count of all words in every doc
                if word not in dict:
                    dict[word] = 1
                else:
                    dict[word] += 1
            tf_idf.append(dict)
        file1.close()

    # calculating final TF-IDF values  for all words in all docs(line in a doc in this case)
    for doc in tf_idf:
        for key in doc:
            true_idf = math.log(len(tf_idf) / idf[key])
            true_tf = doc[key] / len(doc)
            doc[key] = true_tf * true_idf

    # do not get overwhelmed, just for logging the quantity of words that have been processed.
    print(
        TAG,
        "Total number of unique words in corpus",
        len(idf),
        "( " + paint("++" + str(len(idf) - prev_doc_count), "g") + " )"
        if prev_file_path
        else "",
    )
    print(
        TAG,
        "Total number of docs in corpus:",
        len(tf_idf),
        "( " + paint("++" + str(len(tf_idf) - prev_corpus_length), "g") + " )"
        if prev_file_path
        else "",
    )

    # dump if a dir-path is given
    if dump_path:
        if dump_path[-8:] != "tfidfpkl":
            raise Exception(
                TAG
                + "Please provide a .tfidfpkl file_path, it is the standard format of this module."
            )
        pickle.dump(
            (idf, tf_idf), open(dump_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL
        )
        print(TAG, "Dumping TF-IDF vars @", dump_path)
    return idf, tf_idf
