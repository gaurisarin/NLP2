import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertForTokenClassification, DistilBertConfig
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, Linear
from preprocess import convert_to_bio, encoder, padding_func
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from data_loader import MovieQueriesDataset
import logging, sys
import numpy as np
from preprocess import intents, bio_tags
from preprocess import tokenizer

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout
)

model_filepath = 'NER_and_intent_model'

queries_df = pd.read_csv('queries_data.csv')

queries_df[['Intent', 'Tokens', 'BIO_tags']] = queries_df.apply(convert_to_bio, axis=1)
input_ids, attention_masks, intent_labels, bio_labels = encoder(queries_df)

# Splitting the dataset into train, validation, and test sets
train_val_queries, test_queries = train_test_split(queries_df, test_size=0.1, random_state=42)
train_queries, val_queries = train_test_split(train_val_queries, test_size=0.11, random_state=42)

# Preparing datasets for DataLoader
train_dataset = MovieQueriesDataset(
    queries=[input_ids[i] for i in train_queries.index],
    attention_masks=[attention_masks[i] for i in train_queries.index],
    intents=[intent_labels[i] for i in train_queries.index],
    bio_labels=[bio_labels[i] for i in train_queries.index]
)

val_dataset = MovieQueriesDataset(
    queries=[input_ids[i] for i in val_queries.index],
    attention_masks=[attention_masks[i] for i in val_queries.index],
    intents=[intent_labels[i] for i in val_queries.index],
    bio_labels=[bio_labels[i] for i in val_queries.index]
)

test_dataset = MovieQueriesDataset(
    queries=[input_ids[i] for i in test_queries.index],
    attention_masks=[attention_masks[i] for i in test_queries.index],
    intents=[intent_labels[i] for i in test_queries.index],
    bio_labels=[bio_labels[i] for i in test_queries.index]
)

# DataLoader setup
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=padding_func)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=padding_func)
test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=padding_func)

num_intents = len(queries_df['Intent'].unique())
num_entities_labels = len(bio_tags)
num_epoch = 10

config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
config.num_labels = num_entities_labels


def train_model():

    # Load DistilBERT with a token classification head (used for NER)
    model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', config=config)

    # Add a separate classification head for intents
    model.intent_classifier = Linear(config.dim, num_intents)

    # run on GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loss functions
    loss_fct_intent = CrossEntropyLoss()
    loss_fct_ner = CrossEntropyLoss(ignore_index=-100)

    # Optimizer (you can fine-tune these hyperparameters)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # initialize best validation loss value
    best_val_loss = np.inf

    # move the model to the device and set it to train mode
    model.to(device)
    model.train()

    # Training loop
    for i in range(num_epoch):
        total_intent_loss = 0
        total_bio_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            input_ids = batch['input_ids']
            attention_masks = batch['attention_mask']
            intent_labels = batch['intent_labels']
            bio_labels = batch['bio_labels'].view(-1)

            outputs = model(input_ids, attention_masks, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            cls_representation = hidden_states[-1][:, 0, :]
            intent_logits = model.intent_classifier(cls_representation)

            bio_logits = outputs.logits.view(-1, config.num_labels)

            # Calculate loss for both tasks
            intent_loss = loss_fct_intent(intent_logits, intent_labels)
            bio_loss = loss_fct_ner(bio_logits, bio_labels)

            # Combine the losses
            loss = intent_loss + bio_loss

            # losses info for logger
            total_intent_loss += intent_loss.item()
            total_bio_loss += bio_loss.item()

            loss.backward()
            optimizer.step()

        # Calculate average losses for the epoch
        avg_intent_loss = total_intent_loss / len(train_loader)
        avg_bio_loss = total_bio_loss / len(train_loader)
        # Log the average losses for the epoch
        logging.info(
            f"Epoch {i + 1}/{num_epoch} - Avg Intent Loss: {avg_intent_loss:.4f} - Avg BIO Loss: {avg_bio_loss:.4f}")

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_total_intent_loss = 0
            val_total_bio_loss = 0
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                input_ids = batch['input_ids']
                attention_masks = batch['attention_mask']
                intent_labels = batch['intent_labels']
                bio_labels = batch['bio_labels'].view(-1)

                outputs = model(input_ids, attention_masks, output_hidden_states=True)
                hidden_states = outputs.hidden_states

                cls_representation = hidden_states[-1][:, 0, :]
                intent_logits = model.intent_classifier(cls_representation)

                bio_logits = outputs.logits.view(-1, config.num_labels)

                intent_loss = loss_fct_intent(intent_logits, intent_labels)
                bio_loss = loss_fct_ner(bio_logits, bio_labels)

                val_total_intent_loss += intent_loss.item()
                val_total_bio_loss += bio_loss.item()

            avg_val_intent_loss = val_total_intent_loss / len(val_loader)
            avg_val_bio_loss = val_total_bio_loss / len(val_loader)
            total_val_loss = avg_val_intent_loss + avg_val_bio_loss
            logging.info(
                f"Validation - Avg Intent Loss: {avg_val_intent_loss:.4f} - Avg BIO Loss: {avg_val_bio_loss:.4f}")

            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                logging.info(f"Saving the best model with total loss: {best_val_loss:.4f}")
                model.save_pretrained(model_filepath)
                torch.save(model.state_dict(), f'{model_filepath}.pth')
                logging.info(f'Model saved to {model_filepath}')
        model.train()


if __name__ == '__main__':
    train_model()

# Save the model
#model.save_pretrained('NER_and_intent_model')

# Load the model
# model = DistilBertForTokenClassification.from_pretrained('model_filepath')
