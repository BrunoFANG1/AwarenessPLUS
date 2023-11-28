# Import necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader
from xml.etree import ElementTree as ET
from transformers import DistilBertTokenizerFast


# Create the PyTorch Dataset class
class ArticleDataset(Dataset):
    def __init__(self, article_file, label_file):
        # Parse the XML files
        self.articles = self.parse_xml(article_file)
        self.labels = self.parse_xml(label_file)
        
        # Create a dictionary for faster lookup of labels using article IDs
        self.label_dict = {label['id']: label['hyperpartisan'] for label in self.labels}
        
        # Create a mapping for hyperpartisanship to integers
        self.hyperpartisan_to_int = {'true': 1, 'false': 0}

        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

        
    def parse_xml(self, file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        items = []
        for item in root.findall(".//article"):
            article_id = item.get('id')
            if 'hyperpartisan' in item.attrib:
                hyperpartisan = item.get('hyperpartisan')
                items.append({'id': article_id, 'hyperpartisan': hyperpartisan})
            else:
                paragraphs = [p.text for p in item.findall(".//p")]
                items.append({'id': article_id, 'paragraphs': paragraphs})
        return items
    
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        article = self.articles[idx]
        article_id = article['id']
        paragraphs = article['paragraphs']
        
        concatenated_paragraphs = " ".join(filter(None, paragraphs))
        
        hyperpartisan = self.label_dict.get(article_id, 'unknown')
        
        if hyperpartisan != 'unknown':
            hyperpartisan = self.hyperpartisan_to_int[hyperpartisan]

        x_token = self.tokenizer(concatenated_paragraphs,
                                 padding='max_length',
                                 max_length=512,
                                 truncation=True,
                                 return_tensors='pt')

        return {'id': x_token['input_ids'][0], 'attention_mask': x_token['attention_mask'][0], 'labels': hyperpartisan}

        # return {'id': article_id, 'body': concatenated_paragraphs, 'hyperpartisan': hyperpartisan}


# if __name__ == '__main__':
# # Initialize the ArticleDataset
#     article_dataset = ArticleDataset('./hyperpartisan/articles-test-bypublisher-20181212/articles-test-bypublisher-20181212.xml', './hyperpartisan/ground-truth-test-bypublisher-20181212/ground-truth-test-bypublisher-20181212.xml')
#     article_dataloader = DataLoader(article_dataset, batch_size=3, shuffle=True)

#     for batch in article_dataloader:
#         print("Batch:")
#         print("ID: ", batch['id'])
#         print("Article: ", batch['body'])
#         print("hyperpartisan: ", batch['hyperpartisan'])
#         break
