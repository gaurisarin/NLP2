import torch
import random
from model_class import DistilBertForNERAndIntent
from class_training import num_entities_labels, num_intents
from preprocess import intent_encoder, bio_encoder
from align_predictions import align_predictions
from preprocess import tokenizer
from class_training import model_filepath
from filterfunct import returnDirector, returnYear, returnLeadActor, analyze_sentiment, change_sentiment, returnGenres
from similarityFunc import main_get_similar_movies

# Create an object of the DistilBERT model for NER and Intent classification
model = DistilBertForNERAndIntent(num_entities_labels, num_intents)

# Load the state dictionary
model.load_state_dict(torch.load(f'{model_filepath}.pth'))

# use GPU if available and move model to the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def predict_intent_and_entities(query):
    '''
    Function to predict the intent and entities of a query
    :param query: inputted query requesting details about a movie
    :return: intent of a query and entities within the query
    '''

    # Tokenize the query, add special tokens and attention mask
    inputs = tokenizer.encode_plus(
        query,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    # move tensors to GPU if available
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        _, intent_logits, bio_logits, _, _ = model.forward(input_ids=input_ids, attention_mask=attention_mask)

    # Get the predicted intent and entities
    intent_predictions = torch.argmax(intent_logits, dim=1)
    entity_predictions = torch.argmax(bio_logits, dim=1)

    # Decode intent and entity labels
    intent_label = intent_encoder.inverse_transform(intent_predictions)[0]
    entity_labels = bio_encoder.inverse_transform(entity_predictions)

    # Align entity labels with tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    aligned_labels = align_predictions(tokens, entity_labels)

    return intent_label, aligned_labels


def chatbot_implementation():
    print("Welcome to your personalized movie chatbot!\nHow can I help you today?")

    while True:
        # input query from user
        query = input(">> ")

        # exit the chatbot
        if query.lower() == 'exit' or query.lower() == 'quit':
            print("Goodbye!")
            break

        # predict the intent and entities of the query
        intent, entities = predict_intent_and_entities(query)

        user_entity = None
        entity_type = None

        # extract the user entity and entity type
        for entity in entities:
            if entity[1].startswith('B-'):
                user_entity = entity[0]
                entity_type = entity[1][2:]
            elif entity[1].startswith('I-') and user_entity:
                user_entity += ' ' + entity[0]
                entity_type = entity[1][2:]
            else:
                continue

        # if the intent or user entity is not recognized, print an error message. Else, print the recognized
        # intent and entity
        if intent is None or user_entity is None:
            print("Sorry, I didn't understand your query.")
            continue
        else:
            if entity_type == 'GENRE':
                print(f'You asked about {user_entity.lower()} movies.')
            elif entity_type == 'MOVIE_TITLE' and intent == 'Feedback':
                print(f'You gave feedback on {user_entity.title()}.')
            elif entity_type == 'MOVIE_TITLE' and intent == 'Synopsis':
                print(f'You asked about a movie recommendation similar to {user_entity.title()}.')
            else:  # if entity_type == 'MOVIE_TITLE' as default case
                print(f'You asked about {intent.lower()} for {user_entity.title()}.')

        return_message = None

        # return the appropriate response based on the intent
        if intent == 'Director':
            return_message = (returnDirector(user_entity))
        elif intent == 'Release Date':
            return_message = (returnYear(user_entity))
        elif intent == 'Lead Actor':
            return_message = (returnLeadActor(user_entity))
        elif intent == 'Genre':
            return_message = (random.choice(returnGenres(user_entity)))
        elif intent == 'Feedback':
            sentiment = analyze_sentiment(query)
            if entity_type == 'MOVIE_TITLE':
                change_sentiment(sentiment, movie=user_entity)
            elif entity_type == 'GENRE':
                change_sentiment(sentiment, genre=user_entity)

            if sentiment > 0:
                print(f"Thank you for letting me know. I will give more recommendations like this.")
                continue
            elif sentiment < 0:
                print(f"Thank you for letting me know. I will avoid recommending movies like this.")
                continue
        elif intent == 'Synopsis':
            return_message = main_get_similar_movies(user_entity)
        else:
            print(f"I'm sorry, I don't have that information.")
            continue

        print(return_message)


def main():
    chatbot_implementation()


if __name__ == '__main__':
    main()
