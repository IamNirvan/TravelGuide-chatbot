import tensorflow as tf
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import random
import pickle
import json

loaded_model_h5 = tf.keras.models.load_model(f'./models/modelV1.h5')

data = json.loads(open('./data/intents_v2.json').read())
words = pickle.load(open('./data/words.pkl', 'rb'))
classes = pickle.load(open('./data/classes.pkl', 'rb'))


def clean_text(text):
  lemmatizer = WordNetLemmatizer()
  tokens = nltk.word_tokenize(text)
  return [lemmatizer.lemmatize(word) for word in tokens]

def bag_of_words(text, vocab):
  tokens = clean_text(text)
  bow = [0] * len(vocab)

  for token in tokens:
    for index, word in enumerate(vocab):
      if word == token:
        bow[index] = 1
  return bow

def predict_class(text, vocab, labels):
  bow = bag_of_words(text, vocab)
  result = loaded_model_h5.predict(np.array([bow]))[0]
  threshold = 0.5

  y_prediction = [[index, result] for index, result in enumerate(result) if result > threshold]
  y_prediction.sort(key=lambda x: x[1], reverse=True) # Sort by probability in descending order
  result = []

  for prediction in y_prediction:
    result.append(labels[prediction[0]])

  return result

def get_response(intents_list, intents_json):
  if len(intents_list) == 0:
    result = 'Come again?';
  else:
    tag = intents_list[0]

    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
      if intent['tag'] == tag:
        result = random.choice(intent['responses'])
        break
  return result


print('Enter \'exit\' to leave the application\n')

while True:
  user_input = input('')
  if user_input == 'exit':
    break
  intents = predict_class(user_input, words, classes)
  result = get_response(intents, data)
  print(result)