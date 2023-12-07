#!/usr/bin/env python
# coding: utf-8

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

    #Y_true and Y_pred store for epoch
    Y_true = []
    Y_pred = []
    val_acc_batch = []
    
    
    val_accuracy_batch = evaluate.load('accuracy')
    
    label = []
    for batch in dataloader:
        input_ids = batch['id'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        
       
        predictions = output.logits
        predictions = torch.argmax(predictions, dim=1)
        label.append(predictions)
    
    return label

def get_political_perspective(leaning, hyperpartisan):
    return (leaning-1) * (hyperpartisan+1)

def main():
    device = 'cuda'
    batch_size = 1
    train_mask = range(50)
    model_name = 'distilbert-base-uncased'
    leaning_model_dir = './saved_models/leaning/'
    hyperpartisan_model_dir = './saved_models/hyperpartisan/'
    dataset = ArticleDataset(json_file='./M1_output.json')
    dataloader = DataLoader(dataset, shuffle=False, batch_size = 1)

    hyperpartisan_model = AutoModelForSequenceClassification.from_pretrained(hyperpartisan_model_dir)
    hyperpartisan_model.to(device)
    leaning_model = AutoModelForSequenceClassification.from_pretrained(leaning_model_dir)
    leaning_model.to(device)

    leaning_predictions = evaluate_model(leaning_model, dataloader, device)
    hyperpartisan_predictions = evaluate_model(hyperpartisan_model, dataloader, device)

    coarse_perspectives = []
    for i in range(len(leaning_predictions)):
        coarse_perspectives.append(int(get_political_perspective(leaning_predictions[i], hyperpartisan_predictions[i]).data))

    print(coarse_perspectives)

if __name__ == "__main__":
    main()