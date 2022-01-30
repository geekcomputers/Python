#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random


# In[ ]:


# analysing tweets from the corpus


# In[14]:


positive_tweets = twitter_samples.strings("positive_tweets.json")


# In[15]:


negative_tweets = twitter_samples.strings("negative_tweets.json")


# In[16]:


all_tweets = positive_tweets + negative_tweets


# In[17]:


# Analysing sampels tweets

print(positive_tweets[random.randint(0, 5000)])


# In[19]:


""" There are 4 basic steps in pre-processing of any text 
1.Tokenizing
2.Removing hyper links if any
3.Converting to lower case
4.Removing punctuations
5.steeming of the word"""


import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


# In[20]:


# Removing Hyper links

tweet = all_tweets[1]

# removing RT words in the tweet
tweet = re.sub(r"^RT[\s]+", "", tweet)
# removing hyperlinks in the tweet
tweet = re.sub(r"https?:\/\/.*[\r\n]*", "", tweet)
# removing #symbol from the tweet
tweet = re.sub(r"#", "", tweet)

print(tweet)


# In[22]:


# Tokenizing

tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

tokens = tokenizer.tokenize(tweet)

print(tokens)


# In[23]:


# Remving stop words and punctuation marks

stoper = stopwords.words("english")

punct = string.punctuation

print(stoper)
print(punct)


# In[24]:


cleaned = []
for i in tokens:
    if i not in stoper and i not in punct:
        cleaned.append(i)


print(cleaned)


# In[25]:


# stemming

stemmer = PorterStemmer()

processed = []

for i in cleaned:
    st = stemmer.stem(i)
    processed.append(st)

print(processed)


# In[ ]:
