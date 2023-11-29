#!/usr/bin/env python
# coding: utf-8

# newsapi.ai key: 0215dfd8-3e70-4f43-868f-6991882ac84f
# 100 items = 1 token, 2000 total tokens per month

from datetime import datetime as dt 
from datetime import timedelta
from eventregistry import *
from tqdm import tqdm
import json

def main():
    # News in now-timespan will be kept in database 
    now = dt.now()
    timespan = timedelta(days=30)
    start_date = now.date()-timedelta(days=1)
    end_date = now.date()-timedelta(days=1)
    num_articles = 100
    source_percentile = 10  # Top percentage according to Alexa source ranking

    print(f'Database timespan: past {timespan} days')

    er = EventRegistry(apiKey = '0215dfd8-3e70-4f43-868f-6991882ac84f')
    query = {
        "$query": {
            "$and": [
                {
                    "categoryUri": "news/Politics"
                },
                {
                    "dateStart": f"{start_date.strftime('%Y-%m-%d')}",
                    "dateEnd": f"{end_date.strftime('%Y-%m-%d')}",
                    "lang": "eng"
                }
            ]
        },
        "$filter": {
            "startSourceRankPercentile": 0,
            "endSourceRankPercentile": source_percentile,
            "isDuplicate": "skipDuplicates"
        }
    }
    q = QueryArticlesIter.initWithComplexQuery(query)
    # change maxItems to get the number of results that you want
    articles = []
    print(f'Downloading {num_articles} articles from {start_date} to {end_date}:')
    for article in tqdm(q.execQuery(er, maxItems=num_articles)):
        articles.append(article)

    remove_count = 0
    add_count = 0
    print(f'Saving articles to database:')
    with open('news_database_dict.json', 'r') as dict_json_file:
        news_database_dict = json.load(dict_json_file)
    with open('news_database.json', 'r') as data_json_file:
        news_database = json.load(data_json_file)
    for a in articles:
        if a['title'] not in news_database_dict.keys():
            news_database_dict[a['title']] = [a['date'], a['title'], a['source'], a['url'], a['body'], a['image']]
            news_database.append({'date':a['date'], 
                                'title':a['title'], 
                                'source_url':a['source']['uri'], 
                                'source_name':a['source']['title'], 
                                'url':a['url'], 
                                'body':a['body'], 
                                'image_url':a['image']})
            add_count += 1
    print(f'{add_count} new articles added.')
    for key in news_database_dict: # Remove old news
        if dt.strptime(news_database_dict[key][0], '%Y-%m-%d') < now - timespan: 
            news_database[:] = [e for e in news_database if e['title'] != key]
            del news_database_dict[key]
            remove_count += 1
    print(f'{remove_count} outdated articles removed.')
    with open('news_database_dict.json', 'w') as out_dict:
        json.dump(news_database_dict, out_dict, indent=4)
    with open('news_database.json', 'w') as outfile:
        json.dump(news_database, outfile, indent=4)

if __name__ == "__main__":
    main()



