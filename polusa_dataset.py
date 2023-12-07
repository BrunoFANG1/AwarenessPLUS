from torch.utils.data import Dataset
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast

class POLUSADataset(Dataset):
    def __init__(self, csv_file):
        # Read the CSV file
        self.data = pd.read_csv(csv_file)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

        # Create a dictionary to map political leanings to integers
        self.political_leaning_to_int = {'LEFT': 0, 'RIGHT': 1, 'CENTER': 2} #, 'UNDEFINED': 3}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Extract the relevant fields
        id_ = self.data.iloc[idx, 0]
        x_token = self.tokenizer(self.data.iloc[idx, 5],
                                 padding='max_length',
                                 max_length=512,
                                 truncation=True,
                                 return_tensors='pt')
        
        political_leaning = self.data.iloc[idx, 9]
        
        # Convert political leaning to integer
        political_leaning = self.political_leaning_to_int.get(political_leaning)  # Use 3 for 'UNDEFINED' or unknown
        # political_leaning = self.political_leaning_to_int.get(political_leaning, 3)  # Use 3 for 'UNDEFINED' or unknown
        
        return {'id': x_token['input_ids'][0], 'attention_mask': x_token['attention_mask'][0], 'labels': political_leaning}


# if __name__== '__main__':
# 	# Initialize the custom dataset
# 	custom_dataset = CustomDataset('./polusa/2017_1.csv')


# 	from torch.utils.data import DataLoader

# 	# Create a DataLoader
# 	batch_size = 10  # You can change this value based on your needs
# 	dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# 	# Test by getting one batch of data
# 	for batch in dataloader:
# 		print("Sample batch:")
# 		print("ID: ", batch['id'])
# 		print("Article: ", batch['body'])
# 		print("Political Leaning: ", batch['political_leaning'])
# 		break  # Stop after getting one batch
