#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:59:45 2022

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
import wordninja

import scattertext as st
from IPython.display import IFrame
from IPython.core.display import display, HTML
from scattertext import CorpusFromPandas, produce_scattertext_explorer
display(HTML("<style>.container { width:98% !important; }</style>"))


test = pd.read_csv('test.csv')

test = test[["title", "selftext", "author"]]
test["selftext"].fillna("empty",inplace=True)

def processing_text(series_to_process):
    new_list = []
    tokenizer = RegexpTokenizer(r'(\w+)')
    lemmatizer = WordNetLemmatizer()
    for i in range(len(series_to_process)):
        # transfer all the letter to lower case
        dirty_string = (series_to_process)[i].lower()
        #receives a stream of characters, breaks it up into individual tokens (usually individual words), and outputs a stream of tokens
        words_only = tokenizer.tokenize(dirty_string)
        #还原词性,Lemmatization is the process of grouping together the different inflected forms of a word so they can be analyzed as a single item
        words_only_lem = [lemmatizer.lemmatize(i) for i in words_only]
        #删除stop words
        words_without_stop = [i for i in words_only_lem if i not in stopwords.words("english")]

        # join all the world together and save to long_string_clean
        long_string_clean = " ".join(word for word in words_without_stop)
        new_list.append(long_string_clean)
    return new_list

def processing_author_names(series_to_process):
    author_split = []
    for i in range(len(series_to_process)):
        # first we need to split the username in to separate words
        splits_list = wordninja.split(series_to_process[i])
        # join the separate words to a single string, use space separate
        combined_string = " ".join(splits_list)
        author_split.append(combined_string)
    new_list = []
    tokenizer = RegexpTokenizer(r'(\w+)')
    lemmatizer = WordNetLemmatizer()
    for i in range(len(author_split)):
        #小写
        dirty_string = (author_split)[i].lower()
        #分词，去标点
        words_only = tokenizer.tokenize(dirty_string) 
        #还原词性
        words_only_lem = [lemmatizer.lemmatize(i) for i in words_only]
        #去掉stop words
        words_without_stop = [i for i in words_only_lem if i not in stopwords.words("english")]
        #连接
        long_string_clean = " ".join(word for word in words_without_stop)
        new_list.append(long_string_clean)
    return new_list


test["selftext_clean"] = processing_text(test["selftext"])
test["title_clean"] = processing_text(test["title"])
test["author_clean"]= processing_author_names(test["author"])

# combine the 3 data to one colums and 
test["megatext_clean"]=test["author_clean"] + " " + test["selftext_clean"]+ " " +test["title_clean"]

test[["author","title","selftext","megatext_clean"]].to_csv('ready.csv', index = False)


