#!/usr/bin/env python
# coding: utf-8



import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
import evaluate as evaluate
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import copy
import json
from transformers import DistilBertTokenizerFast


class ArticleDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as input_file:
            self.data = json.load(input_file)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def __len__(self):
        return 1
    def __getitem__(self, idx):
        x_token = self.tokenizer(self.data,
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

def label_to_hyperpartisan(label):
    if label == 0:
        return 'false'
    elif label == 1:
        return 'true'
    
def label_to_leaning(label):
    if label == 0:
        return 'left'
    elif label == 1:
        return 'right'
    elif label == 2:
        return 'center'
    elif label == 3:
        return 'undefined'

def main():

    device = 'cuda'

    leaning_model_dir = './saved_models/leaning/'
    hyperpartisan_model_dir = './saved_models/hyperpartisan/'


    dataset = ArticleDataset(json_file='./input/input_article.json')
    dataloader = DataLoader(dataset)


    # leaning_model = AutoModelForSequenceClassification.from_pretrained(leaning_model_dir)
    hyperpartisan_model = AutoModelForSequenceClassification.from_pretrained(hyperpartisan_model_dir)
    hyperpartisan_model.to(device)
    leaning_model = AutoModelForSequenceClassification.from_pretrained(leaning_model_dir)
    leaning_model.to(device)

    leaning_prediction = evaluate_model(leaning_model, dataloader, device)
    hyperpartisan_prediction = evaluate_model(hyperpartisan_model, dataloader, device)

    print(f"Political Leaning: {label_to_leaning(int(leaning_prediction[0].data))}")
    print(f"Is Hyperpartisan: {label_to_hyperpartisan(int(hyperpartisan_prediction[0].data))}")

if __name__ == "__main__":
    main()




