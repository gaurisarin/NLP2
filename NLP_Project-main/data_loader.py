import torch
from torch.utils.data import Dataset


# class to convert the dataset into a format that can be used by our model
class MovieQueriesDataset(Dataset):

    # initialize the class with queries, attention masks, intents, and BIO labels
    def __init__(self, queries, attention_masks, intents, bio_labels):
        self.queries = queries
        self.attention_masks = attention_masks
        self.intents = intents
        self.bio_labels = bio_labels

    # return the length of the dataset
    def __len__(self):
        return len(self.queries)

    # return the data at a specific index
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.queries[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),
            'intent_labels': torch.tensor(self.intents[idx], dtype=torch.long),
            'bio_labels': torch.tensor(self.bio_labels[idx], dtype=torch.long)
        }
