#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:43:49 2022

"""
import requests
import time
import pandas as pd
from random import randint


def reddit_scrape(url_string, number_of_scrapes, output_list):
    #SCRAPED POSTS WILL BE CONTAINED IN OUTPUT LIST(SHD BE EMPTY)
    #THIS IS USEFUL FOR THE FIRST SCRAPE FROM THE VIRGIN SUBREDDIT
    after = None 
    for _ in range(number_of_scrapes):
        if _ == 0:
            print("SCRAPING {}\n--------------------------------------------------".format(url_string))
            print("<<<SCRAPING COMMENCED>>>") 
            print("Downloading Batch {} of {}...".format(1, number_of_scrapes))
        elif (_+1) % 5 ==0:
            print("Downloading Batch {} of {}...".format((_ + 1), number_of_scrapes))
        
        if after == None:
            params = {}
        else:
            #THIS WILL TELL THE SCRAPER TO GET THE NEXT SET AFTER REDDIT'S after CODE
            params = {"after": after}             
        res = requests.get(url_string, params=params, headers=headers)
        if res.status_code == 200:
            the_json = res.json()
            output_list.extend(the_json["data"]["children"])
            after = the_json["data"]["after"]
        else:
            print(res.status_code)
            break
    
    print("<<<SCRAPING COMPLETED>>>")
    print("Number of posts downloaded: {}".format(len(output_list)))
    print("Number of unique posts: {}".format(len(set([p["data"]["name"] for p in output_list]))))
 
def create_unique_list(original_scrape_list, new_list_name):
    data_name_list=[]
    for i in range(len(original_scrape_list)):
        if original_scrape_list[i]["data"]["name"] not in data_name_list:
            new_list_name.append(original_scrape_list[i]["data"])
            data_name_list.append(original_scrape_list[i]["data"]["name"])
    #CHECKING IF THE NEW LIST IS OF SAME LENGTH AS UNIQUE POSTS
    print("LIST NOW CONTAINS {} UNIQUE SCRAPED POSTS".format(len(new_list_name)))



test = [] #DEFINING AN EMPTY LIST THAT WILL CONTAIN OUR SCRAPED DATA
reddit_scrape("https://www.reddit.com/r/SuicideWatch/new.json", 20, test)

final = []
create_unique_list(test, final)
final = pd.DataFrame(final)
final.to_csv('test.csv', index = False)

test = final[["title", "selftext", "author"]]
























    
