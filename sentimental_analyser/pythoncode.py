#punctuation_chars = ["'", '"', ",", ".", "!", ":", ";", '#', '@']
# lists of words to use
punctuation_chars = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '#', 'å', '\x88', '¶', 'æ', '\x9c', '\x8d', 'ã', '\x82', '\x86', 'ã', '\x81', '«', 'ã', '\x81', '°']
def strip_punctuation(string):
    for char in punctuation_chars:
        c = string.replace(char , '')
        if c != string:
            string = c
    return string 

def get_pos(sentence):
    lst_string = sentence.split()
    new_string =[]
    for word in lst_string:
        new_string.append(strip_punctuation(word))
    counter =0
    for words in new_string:
        for pos in positive_words:
           
            if words.lower() == pos:
               
                counter = counter+1
    return counter


def get_neg(sentence):
    lst_string = sentence.split()
    new_string =[]
    for word in lst_string:
        new_string.append(strip_punctuation(word))
    counter =0
    for words in new_string:
        for neg in negative_words:
            if words.lower() == neg:
                
                counter = counter+1
    return counter
positive_words = []
with open("positive-words.txt") as pos_f:
    for lin in pos_f:
        if lin[0] != ';' and lin[0] != '\n':
            positive_words.append(lin.strip())


negative_words = []
with open("negative-words.txt") as pos_f:
    for lin in pos_f:
        if lin[0] != ';' and lin[0] != '\n':
            negative_words.append(lin.strip())



lst_string = []
with open('test.csv', 'r') as twdata:
    twdata = twdata.readlines()
    header = twdata[0]
    field_names = header.strip().split(',')
    for row in twdata[1:]:
        lst_string.append(row.strip().split(','))

for each_tweet in lst_string:
    #print(each_tweet)
    c  = 0
    d = 0
    for sentence in each_tweet[1:]:
        c = get_pos(sentence) + c
        d = get_neg(sentence) + d
    each_tweet.append(c)
    each_tweet.append(d)
    if c > d:
        net_score = c
    else:
        net_score = -d
    each_tweet.append(net_score)
        
#print(lst_string[:4])
outfile = open("resulting_data.csv",'w')
outfile.write('id,tweets, Positive Score, Negative Score, Net Score')
outfile.write('\n')
for fin_lst in lst_string:
    row_string = ""
    for each in fin_lst:
        row_string = row_string +  str(each) + "," 
    outfile.write(row_string)
    outfile.write('\n')
outfile.close()


outfile = open("resulting_.csv",'w')
outfile.write('id, Positive Score, Negative Score, Net Score')
outfile.write('\n')
for fin_lst in lst_string:
    row_string = '{},{},{},{}'.format(fin_lst[0],fin_lst[-3],fin_lst[-2],fin_lst[-1])
    #print(row_string)
    outfile.write(row_string)
    outfile.write('\n')
outfile.close()