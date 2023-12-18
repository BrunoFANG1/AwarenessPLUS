#!/usr/bin/env python
# coding: utf-8
import threading
import time 
import sys
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
from transformers import DistilBertTokenizerFast
import evaluate as evaluate
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import copy
import json
from NewsSentiment import TargetSentimentClassifier
import json
import time 
import nltk
from nltk.tokenize import sent_tokenize
import random
from flask import Flask, request, jsonify
import joblib
import pickle
import sys
from fastapi import FastAPI
import requests
from urllib.parse import quote_plus
from fastapi import Request
app=FastAPI()

# Used for model evaluation in M2.1
class ArticleDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as input_file:
            self.data = json.load(input_file)
        self.articles = []
        self.articles.append(self.data['match']['docs'][0]['body'][0])
        for doc in self.data['response']['docs']:
            self.articles.append(doc['body'][0])
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def __len__(self):
        return len(self.articles)
    def __getitem__(self, idx):
        x_token = self.tokenizer(self.articles[idx],
                                 padding='max_length',
                                 max_length=512,
                                 truncation=True,
                                 return_tensors='pt')        
        return {'id':x_token['input_ids'][0], 'attention_mask':x_token['attention_mask'][0]}


# Used for M2.2, when we need to find the sentiment of a keyword in an article
def m22(input_fname, article_idx, tsc, multiple_articles=False):
    f = open(input_fname, encoding="utf-8")
    data = json.load(f)

    keyword = data['queried_articles'][article_idx]['keyword']

    def find_idx(text, word):
        idx = text.lower().find(word.lower())
        return idx, idx+len(word)

    user_body = data['user_article']['body']

    cluster_body_list = []


    if multiple_articles:
        num_articles = len(data['queried_articles'])
        for i in range(num_articles):
            cluster_body_list.append(data['queried_articles'][i]['body'])
    else:
        num_articles = 1
        cluster_body_list.append(data['queried_articles'][article_idx]['body'])


    user_sentiments = {}
    idxs = find_idx(user_body, keyword)
    try:
        user_sentiments[keyword] = tsc.infer(
            text=user_body,
            target_mention_from=idxs[0],
            target_mention_to=idxs[1]
        )[0]['class_label']
    except:
        user_sentiments[keyword] = 'neutral'

    
    keyword_indeces = []
    for i in range(num_articles):
        keyword_indeces.append(find_idx(cluster_body_list[i], keyword))
    topic_sentiments = []
    for i in range(num_articles):
        try:
            topic_sentiments.append(tsc.infer(
                text=cluster_body_list[i],
                target_mention_from=keyword_indeces[i][0],
                target_mention_to=keyword_indeces[i][1]
            )[0]['class_label'])
        except:
            topic_sentiments.append('neutral')

    output = {}
    output['user_article'] = {
        'title': data['user_article']['title'],
        'keyword': keyword,
        'sentiment': user_sentiments[keyword]
    }
    articles_list = []
    if multiple_articles:
        for i in range(num_articles):
            articles_list.append({
                'title': data['queried_articles'][i]['title'],
                'keyword': keyword,
                'sentiment': topic_sentiments[i]
            })
    else:
        articles_list.append({
            'title': data['queried_articles'][article_idx]['title'],
            'keyword': keyword,
            'sentiment': topic_sentiments[0] 
        })
    output['queried_articles'] = articles_list

    return output 


# Used for M2.1, translates leaning text to numbers
def get_political_perspective(leaning, hyperpartisan):
    return (leaning-1) * (hyperpartisan+1)


# M2.1, returns coarse political perspectives of articles
def evaluate_model(model, dataloader, device, acc_only=True):
    """ Evaluate a PyTorch Model
    :param torch.nn.Module model: the model to be evaluated
    :param torch.utils.data.DataLoader test_dataloader: DataLoader containing testing examples
    :param torch.device device: the device that we'll be training on
    :param bool acc_only: return only accuracy if true, else also return ground truth and pred as tuple
    :return accuracy (also return ground truth and pred as tuple if acc_only=False)
    """
    
    # turn model into evaluation mode
    model.eval()

    label = []
    for batch in dataloader:
        input_ids = batch['id'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        
       
        predictions = output.logits
        predictions = torch.argmax(predictions, dim=1)
        label.append(predictions)
    
    return label


# Used for M1 (Solr)
def get_document_id(solr_url, query):
    """ Get the ID of the document matching the query. """
    encoded_query = quote_plus(query)
    full_url = f"{solr_url}?q={encoded_query}"
    response = requests.get(full_url)

    if response.status_code == 200:
        response_json = response.json()
        docs = response_json.get('response', {}).get('docs', [])
        if docs:
            return docs[0].get('id')  # Assuming the first document is the correct one
        else:
            print("No documents found for the query.")
            return None
    else:
        print(f"Error during search: {response.status_code}")
        return None


# Used for M1 (Solr)
def find_similar_documents(solr_mlt_url, document_id):
    """ Find documents similar to the one with the provided ID. """
    full_url = f"{solr_mlt_url}?q=id:{quote_plus(document_id)}&mlt.fl=body&mlt.boost=true&mlt.interestingTerms=details&mlt.maxdfpct=10&ros=10"
    response = requests.get(full_url)
    print(response)

    # for run in range(100):
    if response.status_code == 200:
        print("save file")
        return response.text  # Return the raw JSON response text

    return None


# M1, get similar articles with Solr
def get_solr_articles(article_title):
    # Define the base URL for Solr
    solr_url = 'http://localhost:8990/solr/liang/select'
    solr_mlt_url = 'http://localhost:8990/solr/liang/mlt'  # URL for MLT queries

    # Query to find the specific document by title
    query = f'title:"{article_title}"'

    # Get the document ID
    document_id = get_document_id(solr_url, query)

    if document_id:
        print(document_id)
        similar_documents_json = find_similar_documents(solr_mlt_url, document_id)

        if similar_documents_json:
            # Write the raw JSON response to a file
            with open('M1_ouput.json', 'w') as file:
                file.write(similar_documents_json)
    
    return similar_documents_json


# Used for getting sentence with keyword
def get_sentence(keywords, article):
    sents_with_keys = []
    ret_sent = ''
    keyword_idx = 0
    while len(sents_with_keys) == 0: 
        if keyword_idx >= len(keywords):
            return ['','']
        keyword = keywords[keyword_idx]
        sentences = sent_tokenize(article)
        for sentence in sentences:
            if keyword in sentence:
                sents_with_keys.append(sentence)
        
        # # return longest sent in sents_with_keys
        # if len(sents_with_keys) != 0:
        #     for s in sents_with_keys:
        #         if len(s) > len(ret_sent):
        #             ret_sent = s
        
        # return any sent in sents_with_keys        
        if len(sents_with_keys) != 0:
            r_num = random.randint(0, len(sents_with_keys) - 1)
            ret_sent = sents_with_keys[r_num]
            ret_sent = ret_sent.replace('\n', '')
            ret_pair = [keyword, ret_sent]

        keyword_idx += 1

    return ret_pair


# Used for getting keywords from Solr
def get_keywords_list(input_list):
    # Create a list of (keyword, value) pairs
    pairs = [(input_list[i], input_list[i+1]) for i in range(0, len(input_list), 2)]
    # Sort the pairs based on value in descending order
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    # Extract the keywords from the sorted pairs
    sorted_keywords = [keyword for keyword, value in sorted_pairs]
    prefix="body:"
    cleaned_keywords = [keyword.replace(prefix, "") for keyword in sorted_keywords]

    return cleaned_keywords


# Used for getting current article from frontend
def get_current_article(url):
    article = NewsPlease.from_url(url)
    if type(article) == dict:
        print("Failed to get current article: No response")
        return None
    elif len(article.title) == 0:
        print("Failed to get current article: Failed to get article title")
        return None
    else:
        print(article.title)
        return(article.title)


# def background():
#     while True:
#         time.sleep(10)


def run_instance(device, leaning_model, hyperpartisan_model, input_fname, output_fname, current_title):
    
    # # No longer needed because we are getting the article title from the frontend
    # url = "" 
    # current_title = get_current_article(url) # Get the title of the current article, used for solr
    
    # TODO: uncomment the following block when solr is ready
    # for i in range(10):
    #     out = get_solr_articles(current_title)
    #     if out is not None:
    #         print("success")
    #         break
    #     time.sleep(0.5)
    

    print('Running M2.1.')

    dataset = ArticleDataset(json_file='./M1_output.json')
    dataloader = DataLoader(dataset, shuffle=False, batch_size = 1)

    time1 = time.time()

    leaning_predictions = evaluate_model(leaning_model, dataloader, device)
    hyperpartisan_predictions = evaluate_model(hyperpartisan_model, dataloader, device)

    coarse_perspectives = []
    for i in range(len(leaning_predictions)):
        coarse_perspectives.append(int(get_political_perspective(leaning_predictions[i], hyperpartisan_predictions[i]).data))
      
    time2 = time.time()
    print(time2-time1)

    with open(input_fname, 'r', encoding='utf-8') as input_file:
        in_dict = json.load(input_file)

    out_dict = {}
    out_dict['user_article'] = {
        'title' : in_dict['match']['docs'][0]['title'][0],
        'source' : in_dict['match']['docs'][0]['source_name'][0],
        'body' : in_dict['match']['docs'][0]['body'][0],
        'url' : in_dict['match']['docs'][0]['url'][0],
        'image_url' : in_dict['match']['docs'][0]['image_url'][0],
        'date' : in_dict['match']['docs'][0]['date'][0],
        'M2.1_perspectives' :  coarse_perspectives[0]
    }

    out_dict['queried_articles'] = []

    # DISPLAYING EACH ONE OF DIFFERENT PERSPECTIVES FIRST
    diff_arr = []
    for i in range(len(coarse_perspectives) - 1):
        diff_arr.append(abs(coarse_perspectives[0] - coarse_perspectives[i+1]))

    diff_range = 0
    for i in range(len(diff_arr)):  
        if diff_arr[i] != 0:
            diff_range += 1

    perspectives_dict = {} # key: perspective, value: list of indices

    for i in range(5):
        perspectives_dict[i-2] = []
    for i in range(len(coarse_perspectives) - 1):

        perspectives_dict[coarse_perspectives[i+1]].append(i)
    
    curr_perspective = 0
    curr_perspective_indeces = [0, 0, 0, 0, 0]
    iter_count = 0
    while iter_count < diff_range:
        if coarse_perspectives[0] == curr_perspective - 2:
            curr_perspective += 1
        if curr_perspective_indeces[curr_perspective] < len(perspectives_dict[curr_perspective-2]):
            out_dict['queried_articles'].append({
            'title' : in_dict['response']['docs'][perspectives_dict[curr_perspective-2][curr_perspective_indeces[curr_perspective]]]['title'][0],
            'source' : in_dict['response']['docs'][perspectives_dict[curr_perspective-2][curr_perspective_indeces[curr_perspective]]]['source_name'][0],
            'body' : in_dict['response']['docs'][perspectives_dict[curr_perspective-2][curr_perspective_indeces[curr_perspective]]]['body'][0],
            'url' : in_dict['response']['docs'][perspectives_dict[curr_perspective-2][curr_perspective_indeces[curr_perspective]]]['url'][0],
            'image_url' : in_dict['response']['docs'][perspectives_dict[curr_perspective-2][curr_perspective_indeces[curr_perspective]]]['image_url'][0],
            'date' : in_dict['response']['docs'][perspectives_dict[curr_perspective-2][curr_perspective_indeces[curr_perspective]]]['date'][0],
            'M2.1_perspectives' : coarse_perspectives[perspectives_dict[curr_perspective-2][curr_perspective_indeces[curr_perspective]]+1]
            })
            curr_perspective_indeces[curr_perspective] += 1
            iter_count += 1
        curr_perspective += 1
        curr_perspective %= 5

    for i in range(len(perspectives_dict[coarse_perspectives[0]])):
        out_dict['queried_articles'].append({
        'title' : in_dict['response']['docs'][perspectives_dict[coarse_perspectives[0]][i]]['title'][0],
        'source' : in_dict['response']['docs'][perspectives_dict[coarse_perspectives[0]][i]]['source_name'][0],
        'body' : in_dict['response']['docs'][perspectives_dict[coarse_perspectives[0]][i]]['body'][0],
        'url' : in_dict['response']['docs'][perspectives_dict[coarse_perspectives[0]][i]]['url'][0],
        'image_url' : in_dict['response']['docs'][perspectives_dict[coarse_perspectives[0]][i]]['image_url'][0],
        'date' : in_dict['response']['docs'][perspectives_dict[coarse_perspectives[0]][i]]['date'][0],
        'M2.1_perspectives' : coarse_perspectives[perspectives_dict[coarse_perspectives[0]][i]+1]
        })

    # keywords = in_dict['interestingTerms']
    keywords = get_keywords_list(in_dict['interestingTerms'])
    sent = get_sentence(keywords, out_dict['user_article']['body'])
    out_dict['user_article']['keyword'] = sent[0]
    out_dict['user_article']['key_sentence'] = sent[1]
    for i in range(len(out_dict['queried_articles'])):
        sent = get_sentence(keywords, out_dict['queried_articles'][i]['body'])
        out_dict['queried_articles'][i]['keyword'] = sent[0]
        out_dict['queried_articles'][i]['key_sentence'] = sent[1]


    with open(output_fname, 'w') as f:
        json.dump(out_dict, f, indent=4)

    print('Done running instance.')
    return out_dict

# def main():
#     device = 'cpu'
#     leaning_model_dir = './saved_models/leaning/'
#     hyperpartisan_model_dir = './saved_models/hyperpartisan/'
#     input_fname = './M1_output.json'
#     output_fname = './M2_output.json'

#     hyperpartisan_model = AutoModelForSequenceClassification.from_pretrained(hyperpartisan_model_dir)
#     hyperpartisan_model.to(device)
#     leaning_model = AutoModelForSequenceClassification.from_pretrained(leaning_model_dir)
#     leaning_model.to(device)


#     # now threading1 runs regardless of user input
#     threading1 = threading.Thread(target=background)
#     threading1.daemon = True
#     threading1.start()
#     print('Type "run" to run. Type "exit" twice to quit.')
#     while True:
#         if input() == 'run':
#             run_instance(device, leaning_model, hyperpartisan_model, input_fname, output_fname)
#         elif input() == 'exit':
#             sys.exit()
#         else:
#             print('wrong input')

device = 'cpu'
leaning_model_dir = './saved_models/leaning/'
hyperpartisan_model_dir = './saved_models/hyperpartisan/'
input_fname = './M1_output.json'
output_fname = './M2_output.json'

hyperpartisan_model = AutoModelForSequenceClassification.from_pretrained(hyperpartisan_model_dir)
hyperpartisan_model.to(device)
leaning_model = AutoModelForSequenceClassification.from_pretrained(leaning_model_dir)
leaning_model.to(device)
tsc = TargetSentimentClassifier()


@app.post('/predict')
async def predict(request: Request):
    data = await request.json()
    article_title = data['text'][0]
    prediction = run_instance(device, leaning_model, hyperpartisan_model, input_fname, output_fname, article_title)
    print(prediction)
    return prediction
    # url = data.get('url')
    
    #print(url)


# @app.get('/predict')
# async def predict():
#     data = request.json
#     print(data)
#     url = data['url']
#     prediction = run_instance(device, leaning_model, hyperpartisan_model, input_fname, output_fname)
#     print(prediction)

@app.post('/analyze_this')
async def analyze_this():
    # Usage: return the index of the article selected by the user from frontend (starting at 0)
    # Format: data['article_idx'] = the article's index
    data = await request.json()
    article_idx = data['article_idx']
    curr_key_sentiment = m22('./M2_output.json', article_idx=article_idx, tsc=tsc, multiple_articles=False)
    print(curr_key_sentiment)
    return curr_key_sentiment

@app.post('/analyze_all')
async def analyze_all():
    # Usage: return the index of the article selected by the user from frontend (starting at 0)
    # Format: data['article_idx'] = the article's index
    data = await request.json()
    article_idx = data['article_idx']
    all_key_sentiment = m22('./M2_output.json', article_idx=article_idx, tsc=tsc, multiple_articles=True)
    print(all_key_sentiment)
    return all_key_sentiment

if __name__ == '__main__':
    # classifier = joblib.load('./pipeline.pkl')
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=True
    )