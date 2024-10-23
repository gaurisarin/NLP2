import pandas as pd
from transformers import DistilBertTokenizerFast
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
import torch

# Load the dataset
queries_df = pd.read_csv('queries_data.csv')

# Initialize the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# define the intents and entity types
intents = ['Director', 'Lead Actor', 'Release Date', 'Plot/Synopsis', 'Genre', 'Feedback']
bio_tags = ['O', 'B-MOVIE_TITLE', 'B-GENRE', 'I-MOVIE_TITLE', 'I-GENRE']

# encoder for intents
intent_encoder = LabelEncoder()
intent_encoder.fit(intents)

# encoder for BIO tags
bio_encoder = LabelEncoder()
bio_encoder.fit(bio_tags)

# Creating a mapping from encoded labels to original labels
label_mapping_intent = {index: label for index, label in enumerate(intent_encoder.classes_)}
label_mapping_bio = {index: label for index, label in enumerate(bio_encoder.classes_)}


def convert_to_bio(row):
    '''
    Function to convert the entity annotations to BIO format
    :param row: takes in a row containing query, intent, and entity annotation
    :return: intent, tokens, and BIO tags as a pandas Series
    '''
    # get query, intent, entity_annotation from the row
    query, intent, entity_annotation = row['Queries'].lower(), row['Intent'], row['Entity']

    # Encode the query, including special tokens
    encoded_input = tokenizer.encode_plus(
        query,
        add_special_tokens=True,        # Add [CLS] and [SEP]
        return_offsets_mapping=True,    # To align BIO tags with subwords
        padding=False,                  # Do not pad the sequences
        truncation=True                 # Truncate the sequences
    )

    # Extract the tokens and offsets
    tokens = encoded_input['input_ids']
    offsets = encoded_input['offset_mapping']

    # Create an array to hold the BIO tags; initialize all as 'O'
    bio_tags = ['O'] * len(tokens)

    # if there is an entity annotation, update the BIO tags
    if pd.notna(entity_annotation):
        # split the entities if there are multiple
        entities = entity_annotation.split(';')
        for entity in entities:
            # if the title of the movie has colons, replace them with '-' to avoid confusion with BIO tags
            num_colon = len(entity.split(':')) - 1
            if num_colon > 1:
                while num_colon > 1:
                    entity = entity.replace(':', ' -', 1)
                    num_colon -= 1

            # split the entity and entity type
            entity, entity_type = entity.split(':')

            # lowercase the entity
            entity = entity.lower()

            # Find the start and end indices of the entity in the query
            start_char = query.find(entity)
            end_char = start_char + len(entity)

            # Update bio_tags based on the character positions of the entity
            entity_started = False
            for idx, (start, end) in enumerate(offsets):
                # Skip special tokens
                if start == end == 0:
                    continue

                # account for tokenizer splitting words to multiple tokens
                if start <= start_char < end:
                    bio_tags[idx] = f"B-{entity_type}"
                    entity_started = True
                elif entity_started and end_char > start:
                    bio_tags[idx] = f"I-{entity_type}"
                else:
                    entity_started = False

    return pd.Series([intent, tokens, bio_tags], index=['Intent', 'Tokens', 'BIO_tags'])


def encoder(df):
    '''
    Function to encode the intents and BIO tags, tokenize the queries, and create attention masks
    :param df: DataFrame containing queries, intents, and BIO tags
    :return: input_ids, attention_masks, intent_labels, and bio_labels as lists
    '''

    # Encode intents using intent_encoder
    df['Intent'] = intent_encoder.fit_transform(df['Intent'])

    # Encode BIO tags with bio_encoder after fitting the encoder
    bio_tags_flat_list = [tag for sublist in df['BIO_tags'].tolist() for tag in sublist]
    bio_encoder.fit(bio_tags_flat_list)

    # Encode BIO tags using bio_encoder
    df['BIO_Labels'] = df['BIO_tags'].apply(lambda tags: bio_encoder.transform(tags))

    # Tokenize and encode the queries and create attention masks
    df['Tokens'] = df['Queries'].apply(lambda s: tokenizer.encode(s, add_special_tokens=True))
    df['Attention_Masks'] = df['Tokens'].apply(lambda tokens: [1] * len(tokens))

    # Convert columns to lists for training
    input_ids = df['Tokens'].tolist()
    attention_masks = df['Attention_Masks'].tolist()
    intent_labels = df['Intent'].tolist()
    bio_labels = df['BIO_Labels'].tolist()

    return input_ids, attention_masks, intent_labels, bio_labels


def padding_func(batch):
    '''
    Function to pad the input_ids, attention_masks, bio_labels, and intent labels in the batch
    :param batch: batch of data from the DataLoader
    :return: dictionary containing input_ids, attention_mask, intent_labels, and bio_labels with padding
    '''
    # Extract input_ids, attention_masks, bio_labels, and intent labels from the batch and add padding
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_masks = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    bio_labels = pad_sequence([item['bio_labels'] for item in batch], batch_first=True, padding_value=-100)
    intent_labels = torch.tensor([item['intent_labels'] for item in batch])

    # Return a new dictionary for the batch
    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'intent_labels': intent_labels,
        'bio_labels': bio_labels
    }