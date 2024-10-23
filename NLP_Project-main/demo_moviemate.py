from unittest.mock import patch
from querying_testing import main


def simulate_chatbot_interaction():
    '''
    This function simulates the interaction with the chatbot by providing a list of queries.
    :return: an output in the console that simulated the interaction with the chatbot.
    '''
    queries = [
            "Who directed Inception?",
            "Give me a movie similar to The Martian",
            "What year did Interstellar release?",
            "What's a good action movie?",
            "What's another action movie I should watch?",
            "I loved Titanic.",
            "Who was the lead actor in Kung Fu Panda?",
            "Exit"
        ]

    # use patch to simulate the user inputs using the above queries list
    with patch('builtins.input', side_effect=queries):
        main()


if __name__ == '__main__':
    simulate_chatbot_interaction()
