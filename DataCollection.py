import requests
import pandas as pd


class DataCollection:

    def __init__(self):

        # empty set to contain the data from reddit
        depress_scraped = [] 
        # call the function to get the data
        self.reddit_scrape("https://www.reddit.com/r/depression.json", 50, depress_scraped)

        depress_scraped_unique = []
        self.create_unique_list(depress_scraped, depress_scraped_unique)

        
        depression = pd.DataFrame(depress_scraped_unique)
        # Add a extra column, because we collect depression data here, so the is_suicide feature should be 0
        depression["is_suicide"] = 0
        depression.head()  

        
        suicide_scraped = [] 
        self.reddit_scrape("https://www.reddit.com/r/SuicideWatch.json", 50, suicide_scraped)
       
        suicide_scraped_unique = []
        self.create_unique_list(suicide_scraped, suicide_scraped_unique)
  
        suicide_watch = pd.DataFrame(suicide_scraped_unique)
        suicide_watch["is_suicide"] = 1
        suicide_watch.head() 

        # save both the depression and suicide data to the data folder
        depression.to_csv('data/depression.csv', index=False)
        suicide_watch.to_csv('data/suicide_watch.csv', index=False)

    def reddit_scrape(self,url_string, number_of_scrapes, output_list):
   
        # set afrer be none
        after = None
        # set the header to adoid to many times get
        headers = {"User-agent": "Yaguang"}
        # loop 50 times to grab the data
        for _ in range(number_of_scrapes):
            # if the first time print the header
            if _ == 0:
                print("SCRAPING {}\n--------------------------------------------------".format(url_string))
                print("<<<SCRAPING COMMENCED>>>")
                print("Downloading Batch {} of {}...".format(1, number_of_scrapes))
            elif (_ + 1) % 5 == 0:
                print("Downloading Batch {} of {}...".format((_ + 1), number_of_scrapes))
            # if the first time loop,set the params be empty
            if after == None:
                params = {}
            else:
                # if not the first time set the after params be the previous return.
                params = {"after": after}
            # set the get request
            res = requests.get(url_string, params=params, headers=headers)
            # if success xxtend the output list
            if res.status_code == 200:
                the_json = res.json()
                output_list.extend(the_json["data"]["children"])
                # set the after be the return data for the next time loop
                after = the_json["data"]["after"]
            else:
                print(res.status_code)
                break
       

        print("<<<SCRAPING COMPLETED>>>")
        print("Number of posts downloaded: {}".format(len(output_list)))
        print("Number of unique posts: {}".format(len(set([p["data"]["name"] for p in output_list]))))


    # reform a unique data list   
    def create_unique_list(self,original_scrape_list, new_list_name):
        data_name_list = []
        for i in range(len(original_scrape_list)):
            if original_scrape_list[i]["data"]["name"] not in data_name_list:
                new_list_name.append(original_scrape_list[i]["data"])
                data_name_list.append(original_scrape_list[i]["data"]["name"])
        
        print("LIST NOW CONTAINS {} UNIQUE SCRAPED POSTS".format(len(new_list_name)))
