#!/usr/bin/env python
# coding: utf-8

# In[63]:


from re import sub, compile
from string import punctuation
from time import process_time

import pandas as pd
from nltk.corpus import stopwords, gutenberg
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from unidecode import unidecode

# Data for testing
emma = gutenberg.words('austen-emma.txt')
example_text = ' '.join(emma)
df = pd.DataFrame(data={'sentences': sent_tokenize(example_text)})
tokenizer = lambda s: clean_text(s).split()

vectorizer = CountVectorizer(encoding='ascii', decode_error='ignore',
                             strip_accents='ascii',
                             tokenizer=tokenizer, lowercase=False,
                             max_df=0.7,
                             min_df=0.0001
                             )
vectorizer.fit(df['sentences'])

STOP_WORDS = stopwords.words('english')
# Remove any charcater is non alphanumeric or space
pattern_cleaning = compile(r'[^\w\s]|\d')
pattern_stop_words = compile(r'\b(' + r'|'.join(STOP_WORDS) + r')\b\s*')
# First remove punctuation and numbers, then remove stop words
remove_punctuation_r = lambda s: sub(pattern_stop_words, '', sub(pattern_cleaning, '', s.lower()))
remove_short_words = lambda s: ' '.join(filter(lambda w: len(w) > 2, s.split()))
# Remove numbers, short words (one or two characters),
# punctuaction, non ascii characers and stop words
clean_text = lambda s: remove_short_words(remove_punctuation_r(s))

# Data cleaning functions

pattern_cleaning = compile(r'[^\w\s]|\d')
pattern_stop_words = compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
pattern_short_words = compile(r'\b[^\s]{0,2}\b')
exclude = punctuation

remove_punctuation_t = lambda s: unidecode(s).translate(str.maketrans('', '', exclude)).lower()
remove_punctuation_r = lambda s: sub(pattern_stop_words, '', sub(pattern_cleaning, '', s.lower()))
remove_stop_words = lambda s: ' '.join([word for word in s.split() if word not in STOP_WORDS])
remove_stop_words_2 = lambda s: sub(pattern_stop_words, '', s)
remove_stop_words_3 = lambda s: ' '.join(filter(lambda w: len(w) > 2 and not w in STOP_WORDS, s.split()))
remove_short_words = lambda s: ' '.join(filter(lambda w: len(w) > 2, s.split()))
remove_short_words_2 = lambda s: sub(pattern_stop_words, '', s)

clean_text_1 = lambda s: remove_short_words_2(remove_punctuation_r(s))
clean_text_2 = lambda s: remove_short_words(remove_punctuation_r(s))
clean_text_3 = lambda s: remove_stop_words(remove_short_words(remove_punctuation_t(s)))
clean_text_4 = lambda s: remove_stop_words_3(remove_punctuation_t(s))
clean_text_5 = lambda s: remove_stop_words_3(remove_punctuation_r(s))

# Comparing data cleaning ways
func = (clean_text_1, clean_text_2, clean_text_3, clean_text_4, clean_text_5)
title = ('Regex and unidecode, loop (short words)',
         'Regex and unidecode, filter (short words)',
         'Translate and unidecode, filter (short words) ,loops (stop words)',
         'Translate and unidecode, filter (short words, stop words)',
         'Regex, loop (short words, stop words)'
         )
for f, t in zip(func, title):
    print('*' * len(t))
    print(t)
    print('*' * len(t))
    t0 = process_time()
    print(df['sentences'].apply(f).head())
    print(f'Time: {process_time() - t0}')
