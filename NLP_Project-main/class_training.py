import sys
from model_class import DistilBertForNERAndIntent
import torch
from torch.optim import AdamW
import numpy as np
import logging
from torch.utils.data import DataLoader
from preprocess import convert_to_bio, encoder, padding_func, bio_tags, queries_df
from data_loader import MovieQueriesDataset
from sklearn.model_selection import train_test_split

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout
)

logging.info('Beginning preprocessing...')

# Initialize a filepath to save the model
model_filepath = 'NER_and_intent_class_model'

# Convert the queries to BIO format, encode the intents, and tokenize the queries
queries_df[['Intent', 'Tokens', 'BIO_tags']] = queries_df.apply(convert_to_bio, axis=1)
input_ids, attention_masks, intent_labels, bio_labels = encoder(queries_df)

# Splitting the dataset into train, validation, and test sets
train_val_queries, test_queries = train_test_split(queries_df, test_size=0.1, random_state=42, stratify=queries_df['Intent'])
train_queries, val_queries = train_test_split(train_val_queries, test_size=0.11, random_state=42, stratify=train_val_queries['Intent'])

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

logging.info(f'Preprocessing complete. Number of intents: {num_intents}, Number of BIO labels: {num_entities_labels}')


def train_model_class(train_set=train_loader, validation_set=val_loader, save_filepath=model_filepath, num_epoch=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use GPU if available

    # create an object of the DistilBERT model for NER and Intent classification
    model = DistilBertForNERAndIntent(num_entities_labels, num_intents)
    model.to(device)

    # Optimizer - we used AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # initialize the best validation loss value to infinity
    best_val_loss = np.inf

    logging.info('Model Initialized. Starting training now...')

    for i in range(num_epoch):
        # Set the model to training mode
        model.train()

        # Initialize the total loss for the epoch
        total_loss = 0
        total_intent_loss = 0
        total_bio_loss = 0

        # Iterate over the training set batches
        for batch in train_set:
            # Move the batch to the device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            loss, _, _, intent_loss, bio_loss = model.forward(input_ids=batch['input_ids'],
                                                              attention_mask=batch['attention_mask'],
                                                              intent_labels=batch['intent_labels'],
                                                              bio_labels=batch['bio_labels'])

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Update the loss for entities, intents, and the total loss
            total_intent_loss += intent_loss.item()
            total_bio_loss += bio_loss.item()
            total_loss += loss.item()

        # Calculate average losses for the epoch
        avg_intent_loss = total_intent_loss / len(train_set)
        avg_bio_loss = total_bio_loss / len(train_set)

        # Log the average losses for the epoch
        logging.info(
            f"Epoch {i + 1}/{num_epoch} - Avg Intent Loss: {avg_intent_loss:.4f} - Avg BIO Loss: {avg_bio_loss:.4f}")

        # Validation loop - set the model to evaluation mode for validation
        model.eval()

        # use torch.no_grad() to disable gradient computation during validation
        with torch.no_grad():
            # Initialize the total loss for validation
            val_total_intent_loss = 0
            val_total_bio_loss = 0

            # Iterate over the validation set batches
            for batch in validation_set:
                # Move the batch to the device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass
                loss, _, _, val_intent_loss, val_bio_loss = model.forward(input_ids=batch['input_ids'],
                                                                          attention_mask=batch['attention_mask'],
                                                                          intent_labels=batch['intent_labels'],
                                                                          bio_labels=batch['bio_labels'])

                # Update the loss for entities and intents
                val_total_intent_loss += val_intent_loss.item()
                val_total_bio_loss += val_bio_loss.item()
                total_loss += loss.item()

            # Calculate average losses for validation set
            avg_val_intent_loss = val_total_intent_loss / len(validation_set)
            avg_val_bio_loss = val_total_bio_loss / len(validation_set)
            total_val_loss = avg_val_intent_loss + avg_val_bio_loss

            # Log the average losses for the validation set
            logging.info(
                f"Validation - Avg Intent Loss: {avg_val_intent_loss:.4f} - Avg BIO Loss: {avg_val_bio_loss:.4f}")

            # Save the model if the validation loss is the best so far
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                logging.info(f"Saving the best model with total loss: {best_val_loss:.4f}")
                torch.save(model.state_dict(), f'{save_filepath}.pth')
                logging.info(f'Model saved to {save_filepath}.pth')


if __name__ == '__main__':
    train_model_class()
    logging.info(f'Training complete. Model saved to {model_filepath}.')
