import logging
import torch
from sklearn.metrics import classification_report
from model_class import DistilBertForNERAndIntent
from class_training import num_entities_labels, num_intents, test_loader
from preprocess import bio_encoder, label_mapping_bio, label_mapping_intent

# Create an instance of the DistilBERT model for NER and Intent classification
model = DistilBertForNERAndIntent(num_entities_labels, num_intents)

# Load the state dictionary
model.load_state_dict(torch.load('NER_and_intent_class_model.pth'))

# use GPU if available and move model to the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.eval()  # Set model to evaluation mode

# Initialize lists to store the predictions and labels
all_intent_preds = []
all_intent_labels = []
all_entity_preds = []
all_entity_labels = []

# Log the start of the evaluation
logging.info("Starting model evaluation on test data...")

# use torch.no_grad to disable gradient computation
with torch.no_grad():
    # Iterate over the test set
    for batch in test_loader:
        # Move the batch to the device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Extract the input_ids, attention_mask, intent_labels, and bio_labels from the batch
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        intent_labels, bio_labels = batch['intent_labels'].cpu().numpy(), batch['bio_labels'].cpu().view(-1).numpy()

        # Forward pass
        loss, intent_logits, bio_logits, val_intent_loss, val_bio_loss = model(input_ids=batch['input_ids'],
                                                          attention_mask=batch['attention_mask'],
                                                          intent_labels=batch['intent_labels'],
                                                          bio_labels=batch['bio_labels'])

        # Decode intent prediction
        intent_preds = torch.argmax(intent_logits, dim=1).cpu().numpy()
        all_intent_preds.extend(intent_preds)
        all_intent_labels.extend(intent_labels)

        # Decode NER predictions
        entity_preds = torch.argmax(bio_logits, dim=1).cpu().numpy()
        valid_indices = bio_labels != -100  # Ignore padding tokens
        all_entity_preds.extend(entity_preds[valid_indices])
        all_entity_labels.extend(bio_labels[valid_indices])

all_entity_labels = bio_encoder.inverse_transform(all_entity_labels)
all_entity_preds = bio_encoder.inverse_transform(all_entity_preds)

# Calculate and print metrics
intent_report = classification_report(all_intent_labels, all_intent_preds, target_names=[label_mapping_intent[i] for i in range(len(label_mapping_intent))], zero_division=0)
print("Intent Classification Metrics:")
print(intent_report)

entity_report = classification_report(all_entity_labels, all_entity_preds, target_names=[label_mapping_bio[i] for i in range(len(label_mapping_bio))], zero_division=0)
print("Entity Recognition Metrics:")
print(entity_report)

