#!/usr/bin/env python
# coding: utf-8

# In[1]:


from newsplease import NewsPlease
import sys
import json


def main(argv):
    article = NewsPlease.from_url(argv[0])
    print(article.maintext)
    if len(article.maintext) == 0:
        print("Failed to get current article")
    else:
        with open('input_article.json', 'w') as outfile:
            json.dump(article.maintext, outfile, indent = 4)

if __name__ == "__main__":
    main(sys.argv[1:])
