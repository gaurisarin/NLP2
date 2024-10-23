def align_predictions(tokens, predictions):
    '''
    Align the predictions with the tokens
    :param tokens: initial list of tokens inputted to the model
    :param predictions: list of predictions made by the model
    :return: a list of tuples containing the token and its corresponding entity prediction
    '''

    # Initialize lists to store the real tokens and predictions
    real_tokens = []
    real_predictions = []

    # Iterate over the tokens and predictions
    for token, label in zip(tokens, predictions):
        # Remove predictions for special tokens [CLS] and [SEP] and any padding tokens
        if token in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if token.startswith("##"):
            # If the current token is a subword, append it to the previous real token
            if real_tokens:
                real_tokens[-1] += token[2:]
        else:
            # For a new real token, append both the token and its prediction to their respective lists
            real_tokens.append(token)
            real_predictions.append(label)

    # Align the real tokens with their corresponding predictions
    aligned_predictions = list(zip(real_tokens, real_predictions))
    return aligned_predictions
