import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os
import pickle
nltk.download("punkt")
nltk.download("wordnet")

workingDir = './'
filePath = f'{workingDir}/data/intents_v2.json'
print(os.listdir(workingDir))

# Read intents file
intents = open(filePath).read()
data = json.loads(intents)
# print(data)


# Clean data
words = [] # list to store all the tokenized words in the patterns
classes = [] # list to store the tag of a corresponding pattern
data_X = [] # list to store patterns
data_y = [] # list to store tag of associated with each pattern in data_X

for intent in data['intents']:
  for pattern in intent['patterns']:
    tokens = nltk.word_tokenize(pattern) # tokenize the pattern
    words.extend(tokens) # Append tokens into list
    data_X.append(pattern)
    data_y.append(intent['tag'])

  if intent['tag'] not in classes:
    classes.append(intent['tag'])

# print(words)
# print(classes)
# print(data_X)
# print(data_y)

# lemmatize all tokens
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
words = sorted(set(words))
classes = sorted(set(classes))
print(words)
print(classes)

pickle.dump(words, open('./data/words.pkl', 'wb'))
pickle.dump(classes, open('./data/classes.pkl', 'wb'))

# Create training data (BoW)
training = []
out_empty = [0] * len(classes)

for index, doc in enumerate(data_X):
  bag_of_words = []
  text = lemmatizer.lemmatize(doc.lower())

  for word in words:
    bag_of_words.append(1) if word in text else bag_of_words.append(0)

  output_row = list(out_empty)
  # Get the tag for the current pattern. Then get its index.
  # Then in the output_row list, set the value at that index to 1. The rest will be 0
  output_row[classes.index(data_y[index])] = 1
  training.append([bag_of_words, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))
# print(training)

# Implement neural network
model = Sequential()
model.add(Dense(128, input_shape=(len(train_X[0]), ), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
adam = tf.keras.optimizers.Adam(learning_rate = 0.01, decay=1e-6)
model.compile(loss = 'categorical_crossentropy',
              optimizer = adam,
              metrics = ['accuracy']
              )
print(model.summary())
model.fit(x=train_X, y=train_y, epochs=150, verbose=1)

model.save(f'./models/modelV1.h5')