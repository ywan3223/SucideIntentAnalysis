#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 11:36:13 2022

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)




model_data = pd.read_csv('../data/data_for_model.csv', keep_default_na=False)

print(model_data['is_suicide'].mean())

df_list=[]



def multi_modelling(columns_list, model):
    for i in columns_list:

        X = model_data[i]
        y = model_data['is_suicide']
        
  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        
        cvec = CountVectorizer()
        cvec.fit(X_train)
        
  
        X_train = pd.DataFrame(cvec.transform(X_train).todense(),
                               columns=cvec.get_feature_names())
        X_test = pd.DataFrame(cvec.transform(X_test).todense(),
                               columns=cvec.get_feature_names())
        
   
        nb = MultinomialNB()
        nb.fit(X_train,y_train)
        
    
        pred = nb.predict(X_test)
        
      
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        
        
        nb.predict_proba(X_test)
        pred_proba = [i[1] for i in nb.predict_proba(X_test)] 
        auc = roc_auc_score(y_test, pred_proba)

       
        classi_dict = (classification_report(y_test,pred, output_dict=True))

     
        model_results = {}
        model_results['series used (X)'] = i
        model_results['model'] = model
        model_results['AUC Score'] = auc
        model_results['precision']= classi_dict['weighted avg']['precision']
        model_results['recall (sensitivity)']= classi_dict['weighted avg']['recall']
        model_results['confusion matrix']={"TP": tp,"FP":fp, "TN": tn, "FN": fn}
        model_results['train accuracy'] = nb.score(X_train, y_train)
        model_results['test accuracy'] = nb.score(X_test, y_test)
        model_results['baseline accuracy']=0.496
        model_results['specificity']= tn/(tn+fp)  
        model_results['f1-score']= classi_dict['weighted avg']['f1-score']
      
        model_results
        df_list.append(model_results) 

    pd.set_option("display.max_colwidth", 50)
    return (pd.DataFrame(df_list)).round(2)


columns_list = ['selftext', "author", "title",'selftext_clean', "author_clean", "title_clean", "megatext_clean"]
model = "CountVec + MultinomialNB"

multi_modelling(columns_list, model)
test=pd.DataFrame(data=df_list)
test.to_csv('../form/colum_data_.csv', index = False)



def gridsearch_multi(steps_titles, steps_list, pipe_params):

    X = model_data["megatext_clean"]
    y = model_data['is_suicide']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    gs_results = pd.DataFrame(columns=['model','AUC Score', 'precision', 'recall (sensitivity)', 
                                       'best_params', 'best score', 'confusion matrix', 
                                       'train_accuracy','test_accuracy','baseline_accuracy',
                                       'specificity', 'f1-score'])

    for i in range(len(steps_list)):
      
        pipe = Pipeline(steps=steps_list[i])

        gs = GridSearchCV(pipe, pipe_params[i], cv=3) 
        gs.fit(X_train, y_train)
        
      
        pred = gs.predict(X_test)
 
        tn, fp, fn, tp = confusion_matrix(y_test, gs.predict(X_test)).ravel() 
        

        classi_dict = (classification_report(y_test,pred, output_dict=True))
  
        gs.predict_proba(X_test)
        pred_proba = [i[1] for i in gs.predict_proba(X_test)] 
        auc = roc_auc_score(y_test, pred_proba)
        model_results = {}
        model_results['model'] = steps_titles[i]
        model_results['AUC Score'] = auc
        model_results['precision']= classi_dict['weighted avg']['precision']
        model_results['recall (sensitivity)']= classi_dict['weighted avg']['recall']
        model_results['best params'] = gs.best_params_
        model_results['best score'] = gs.best_score_
        model_results['confusion matrix']={"TP": tp,"FP":fp, "TN": tn, "FN": fn}
        model_results['train accuracy'] = gs.score(X_train, y_train)
        model_results['test accuracy'] = gs.score(X_test, y_test)
        model_results['baseline accuracy'] = 0.496
        
        model_results['specificity']= tn/(tn+fp)  
        model_results['f1-score']= classi_dict['weighted avg']['f1-score']


    
        df_list.append(model_results) 
        pd.set_option("display.max_colwidth", 200)
    return (pd.DataFrame(df_list)).round(2)


df_list=[]


steps_titles = ['cvec+ multi_nb','cvec + ss + knn','cvec + ss + logreg']

steps_list = [ 
    [('cv', CountVectorizer()),('multi_nb', MultinomialNB())], #不考虑语法、词的顺序，只考虑所有的词的出现频率 + 朴素贝叶斯
    [('cv', CountVectorizer()),('scaler', StandardScaler(with_mean=False)),('knn', KNeighborsClassifier())], 
    [('cv', CountVectorizer()),('scaler', StandardScaler(with_mean=False)),('logreg', LogisticRegression())]
]


pipe_params = [
    {'cv__stop_words':['english'], 'cv__ngram_range':[(1,1),(1,2)],'cv__max_features': [20, 30, 50],'cv__min_df': [2, 3],'cv__max_df': [.2, .25, .3]},
    {'cv__stop_words':['english'], 'cv__ngram_range':[(1,1),(1,2)],'cv__max_features': [20, 30, 50],'cv__min_df': [2, 3],'cv__max_df': [.2, .25, .3]},
    {'cv__stop_words':['english'], 'cv__ngram_range':[(1,1),(1,2)],'cv__max_features': [20, 30, 50],'cv__min_df': [2, 3],'cv__max_df': [.2, .25, .3]}
]   


gridsearch_multi(steps_titles, steps_list, pipe_params)


steps_titles = ['tvec + multi_nb','tvec + ss + knn','tvec + ss + logreg']


steps_list = [ 
    [('tv', TfidfVectorizer()),('multi_nb', MultinomialNB())],
    [('tv', TfidfVectorizer()),('scaler', StandardScaler(with_mean=False)),('knn', KNeighborsClassifier())], 
    [('tv', TfidfVectorizer()),('scaler', StandardScaler(with_mean=False)),('logreg', LogisticRegression())]
]


pipe_params = [
    {'tv__stop_words':['english'], 'tv__ngram_range':[(1,1),(1,2)],'tv__max_features': [20, 30, 50],'tv__min_df': [2, 3],'tv__max_df': [.2, .25, .3]},
    {'tv__stop_words':['english'], 'tv__ngram_range':[(1,1),(1,2)],'tv__max_features': [20, 30, 50],'tv__min_df': [2, 3],'tv__max_df': [.2, .25, .3]},
    {'tv__stop_words':['english'], 'tv__ngram_range':[(1,1),(1,2)],'tv__max_features': [20, 30, 50],'tv__min_df': [2, 3],'tv__max_df': [.2, .25, .3]}
]   


gridsearch_multi(steps_titles, steps_list, pipe_params)


steps_titles = ['hvec + multi_nb','hvec + ss + knn','hvec + ss + logreg']

steps_list = [ 
    [('hv', HashingVectorizer(alternate_sign=False)),('multi_nb', MultinomialNB())],
    [('hv', HashingVectorizer(alternate_sign=False)),('scaler', StandardScaler(with_mean=False)),('knn', KNeighborsClassifier())], 
    [('hv', HashingVectorizer(alternate_sign=False)),('scaler', StandardScaler(with_mean=False)),('logreg', LogisticRegression())]
]


pipe_params = [
    {'hv__stop_words':['english'], 'hv__ngram_range':[(1,1),(1,2)]},
    {'hv__stop_words':['english'], 'hv__ngram_range':[(1,1),(1,2)]},
    {'hv__stop_words':['english'], 'hv__ngram_range':[(1,1),(1,2)]}
]   


gridsearch_multi(steps_titles, steps_list, pipe_params)


steps_titles = ['hvec + multi_nb (tuning_1)','tvec + multi_nb (tuning_1)']


steps_list = [ 
    [('hv', HashingVectorizer(alternate_sign=False)),('multi_nb', MultinomialNB())],
    [('tv', TfidfVectorizer()),('multi_nb', MultinomialNB())]
]

pipe_params = [
    {'hv__stop_words':['english'], 'hv__ngram_range':[(1,1)], 'hv__n_features': [1000, 1200, 1400, 2000]},
    {'tv__stop_words':['english'], 'tv__ngram_range':[(1,1),(1,2),(1,3)],'tv__max_features': [60, 65, 70, 75, 80],'tv__min_df': [1, 2, 3],'tv__max_df': [.4, .45,.5,.55, .6]},
]   


gridsearch_multi(steps_titles, steps_list, pipe_params)

test=pd.DataFrame(data=df_list)
test.to_csv('../form/data_model.csv', index = False)


X = model_data["megatext_clean"]
y = model_data['is_suicide']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

tvec_optimised = TfidfVectorizer(max_df= 0.5, max_features=70, min_df=2, ngram_range=(1, 3),stop_words = 'english')
X_train_tvec = tvec_optimised.fit_transform(X_train).todense()
X_test_tvec = tvec_optimised.transform(X_test).todense()

nb = MultinomialNB()
nb.fit(X_train_tvec, y_train)
accuracy = nb.score(X_test_tvec, y_test)


pred_proba = [i[1] for i in nb.predict_proba(X_test_tvec)] 
auc = roc_auc_score(y_test, pred_proba)

print("ACCURACY: {}\nAUC SCORE: {}".format(accuracy, auc))

def TF_IDF_most_used_words(category_string, data_series, palette):
    tvec_optimised = TfidfVectorizer(max_df= 0.5, max_features=70, min_df=2, ngram_range=(1, 3),stop_words = 'english')
    tvec_optimised.fit(data_series)
    
    created_df = pd.DataFrame(tvec_optimised.transform(data_series).todense(),
                              columns=tvec_optimised.get_feature_names())
    total_words = created_df.sum(axis=0)
    
    top_20_words = total_words.sort_values(ascending = False).head(20)
    top_20_words_df = pd.DataFrame(top_20_words, columns = ["count"])
    sns.set_style("white")
    plt.figure(figsize = (15, 8), dpi=300)
    ax = sns.barplot(y= top_20_words_df.index, x="count", data=top_20_words_df, palette = palette)
    plt.xlabel("Count", fontsize=9)
    plt.ylabel('Common Words in {}'.format(category_string), fontsize=9)
    plt.yticks(rotation=-5)
    plt.savefig('../images/.most_used_wordsjpg')

TF_IDF_most_used_words("Words used by production model to identify SuicideWatch Posts", model_data["megatext_clean"], "vlag_r")





test_data = pd.read_csv('../test/ready.csv', keep_default_na=False)

X = test_data["megatext_clean"]
X_test_tvec = tvec_optimised.transform(X).todense()
test_pred = nb.predict(X_test_tvec)


test_data['predict'] = test_pred
test_data.to_csv('../test/result.csv', index = False)












