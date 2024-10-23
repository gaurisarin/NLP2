import torch.nn as nn
from transformers import DistilBertForTokenClassification, DistilBertConfig, AdamW
from torch.nn import CrossEntropyLoss, Linear


class DistilBertForNERAndIntent(nn.Module):
    '''
    Class for the DistilBERT model for Named Entity Recognition and Intent Classification
    '''
    def __init__(self, num_entity_labels, num_intent_labels):
        super(DistilBertForNERAndIntent, self).__init__()
        # Initialize the DistilBERT configuration
        config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
        config.num_labels = num_entity_labels

        # Load the pretrained DistilBERT model and add a token classification head
        self.distilbert = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', config=config)

        # Add a separate classification head for intents using a linear layer
        self.intent_classifier = Linear(config.dim, num_intent_labels) # Linear layer for intent classification

        # Initialize loss functions
        self.loss_fct_intent = CrossEntropyLoss()
        self.loss_fct_ner = CrossEntropyLoss(ignore_index=-100) # -100 index is ignored because it is padding token

    def forward(self, input_ids, attention_mask, intent_labels=None, bio_labels=None):
        # get the outputs from the distilbert model
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # get the hidden states from outputs
        hidden_states = outputs.hidden_states

        # get the cls token representation
        cls_representation = hidden_states[-1][:, 0, :]

        # get the logits for the intent classification
        intent_logits = self.intent_classifier(cls_representation)

        # get the logits for the NER
        bio_logits = outputs.logits.view(-1, self.distilbert.config.num_labels)

        # Initialize loss values to none
        loss = None
        intent_loss = None
        bio_loss = None

        # Calculate loss if labels are provided
        if intent_labels is not None and bio_labels is not None:
            # Calculate intent loss
            intent_loss = self.loss_fct_intent(intent_logits.view(-1, self.intent_classifier.out_features),
                                               intent_labels.view(-1))

            # Calculate NER loss
            bio_loss = self.loss_fct_ner(bio_logits.view(-1, self.distilbert.config.num_labels), bio_labels.view(-1))

            # Combine the losses
            loss = intent_loss + bio_loss

        # Return values needed for backpropagation and evaluation
        return loss, intent_logits, bio_logits, intent_loss, bio_loss
