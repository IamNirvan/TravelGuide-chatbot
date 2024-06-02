import tensorflow as tf
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import random
import pickle
import json
import signal
import os

from Util import Util

class Main:
    
    def clean_text(self, text):
        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(text)
        return [lemmatizer.lemmatize(word) for word in tokens]


    def bag_of_words(self, text, vocab):
        tokens = self.clean_text(text)
        bow = [0] * len(vocab)

        for token in tokens:
            for index, word in enumerate(vocab):
                if word == token:
                    bow[index] = 1
        return bow


    def predict_class(self, text, vocab, labels):
        bow = self.bag_of_words(text, vocab)
        result = loaded_model_h5.predict(np.array([bow]))[0]
        threshold = 0.5

        y_prediction = [[index, result] for index, result in enumerate(result) if result > threshold]
        y_prediction.sort(key=lambda x: x[1], reverse=True) # Sort by probability in descending order
        result = []

        for prediction in y_prediction:
            result.append(labels[prediction[0]])

        return result


    def get_response(self, intents_list, intents_json):
        if len(intents_list) == 0:
            result = "I'm sorry, but I am having trouble understanding you..."
        else:
            tag = intents_list[0]

            list_of_intents = intents_json['intents']
            for intent in list_of_intents:
                if intent['tag'] == tag:
                    result = random.choice(intent['responses'])
                    break
        return result


if __name__ == '__main__':
    loaded_model_h5 = tf.keras.models.load_model(f'./models/modelV1.h5')

    data = json.loads(open('./data/intents_v3.json').read())
    words = pickle.load(open('./data/words.pkl', 'rb'))
    classes = pickle.load(open('./data/classes.pkl', 'rb'))

    main = Main()

    # Handle termination signals
    signal.signal(signal.SIGINT, Util.handleSignal)
    signal.signal(signal.SIGTERM, Util.handleSignal)

    print('Enter ctrl + c or \'exit\' to leave the application\n')
    while True:
        user_input = input('')
        if user_input.lower() == 'exit':
            break

        intents = main.predict_class(user_input, words, classes)
        result = main.get_response(intents, data)
        print(result)
