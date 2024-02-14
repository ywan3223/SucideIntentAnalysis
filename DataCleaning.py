#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 11:36:13 2022


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

# read both depression and suicide data
depression = pd.read_csv('../data/depression.csv')
suicide_watch = pd.read_csv('../data/suicide_watch.csv')

# because we have so many columns, but only few of them be relavent,We need to extract the relevant data
depression[["title", "selftext", "author",  "num_comments", "is_suicide","url"]].head()


print(suicide_watch[["title", "selftext", "author",  "num_comments", "is_suicide","url"]].head())

# save the relavent columns to dep_columns and sui_columns
dep_columns = depression[["title", "selftext", "author",  "num_comments", "is_suicide","url"]]
sui_columns = suicide_watch[["title", "selftext", "author",  "num_comments", "is_suicide","url"]]
# because we need clean both depression and suicide_watch, combine the two data togher
combined_data = pd.concat([dep_columns,sui_columns],axis=0, ignore_index=True)
# and save the data to the data folder
combined_data.to_csv('data/combined_data.csv', index = False)

# Start Data Cleaning

# first we need fill the NA in our selftext colum, because someone only post the title but dont have article body
combined_data["selftext"].fillna("empty",inplace=True)

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

# add three new colums
combined_data["selftext_clean"] = processing_text(combined_data["selftext"])
combined_data["title_clean"] = processing_text(combined_data["title"])
combined_data["author_clean"]= processing_author_names(combined_data["author"])


# separate the suicide&depression post, titles and username
suicide_posts = combined_data[combined_data["is_suicide"] ==1]["selftext_clean"]
suicide_titles = combined_data[combined_data["is_suicide"] ==1]["title_clean"]
suicide_authors = combined_data[combined_data["is_suicide"] ==1]["author_clean"]

depression_posts = combined_data[combined_data["is_suicide"] ==0]["selftext_clean"]
depression_titles = combined_data[combined_data["is_suicide"] ==0]["title_clean"]
depression_authors = combined_data[combined_data["is_suicide"] ==0]["author_clean"]


# plot the most used words in the above 6 data and generate 6 images
def plot_most_used_words(category_string, data_series, palette):

    cvec = CountVectorizer(stop_words='english')
    cvec.fit(data_series)

    created_df = pd.DataFrame(cvec.transform(data_series).todense(),
                              columns=cvec.get_feature_names_out())
    total_words = created_df.sum(axis=0)

    top_20_words = total_words.sort_values(ascending = False).head(20)
    top_20_words_df = pd.DataFrame(top_20_words, columns = ["count"])
    sns.set_style("white")
    plt.figure(figsize = (15, 8), dpi=300)
    ax = sns.barplot(y= top_20_words_df.index, x="count", data=top_20_words_df, palette = palette)
    plt.title('\nTop Words used in {}'.format(category_string), fontsize=22)
    plt.xlabel("Count", fontsize=9)
    plt.ylabel('Common Words in {}'.format(category_string), fontsize=9)
    plt.yticks(rotation=-5)
    plt.savefig('images/'+category_string+'.jpg')
    
plot_most_used_words("SuicideWatch Posts", suicide_posts, palette="magma")    
plot_most_used_words("SuicideWatch Titles", suicide_titles, palette="magma")    
plot_most_used_words("SuicideWatch Usernames", suicide_authors, palette="magma")

plot_most_used_words("depression Pots", depression_posts, palette="ocean")
plot_most_used_words("depression Titles", depression_titles, palette="ocean")
plot_most_used_words("depression Usernames", depression_authors, palette="ocean")


# cal the length of the selftext and title
combined_data["selftext_length"]= [len(combined_data["selftext"][i]) for i in range(len(combined_data))]
combined_data["title_length"]= [len(combined_data["title"][i]) for i in range(len(combined_data))]

# combine the 3 data to one colums and 
combined_data["megatext_clean"]=combined_data["author_clean"] + " " + combined_data["selftext_clean"]+ " " +combined_data["title_clean"]

scatter_data = combined_data[["megatext_clean", "is_suicide"]]
scatter_data["category"] = scatter_data["is_suicide"].map({0: "Depression", 1: "Suicide"})

nlp = st.whitespace_nlp_with_sentences
scatter_data.groupby("category").apply(lambda x: x.megatext_clean.apply(lambda x: len(x.split())).sum())
scatter_data['parsed'] = scatter_data.megatext_clean.apply(nlp)


corpus = st.CorpusFromParsedDocuments(scatter_data, category_col="category", parsed_col="parsed").build()


html = produce_scattertext_explorer(corpus,
                                    category='Depression',
                                    category_name='Depression',
                                    not_category_name='Suicide',
                                    width_in_pixels=1000,
                                    jitter=0.1,
                                    minimum_term_frequency=5,
                                    transform=st.Scalers.percentile,
                                    metadata=scatter_data['category']
                                   )
file_name = 'Reddit_ScattertextRankDataJitter.html'
open(file_name, 'wb').write(html.encode('utf-8'))
IFrame(src=file_name, width = 1200, height=700)

combined_data.to_csv('../data/data_for_model.csv', index = False)

















