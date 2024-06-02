from Util import Util
import json
import tensorflow as tf
from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import numpy as np
import random
import pickle
from nltk.stem import WordNetLemmatizer
import string
import nltk
nltk.download("punkt")
nltk.download("wordnet")

class ModelV1:

    def clean_data(self, data: any):
        """
        Tokenizes the patterns and stores the tag associated with each pattern in the
        data_X and data_y lists respectively.
        Returns the tokenized words, classes, data_X and data_y lists.
        """	
        
        try:
            words = [] # stores all the tokenized words in each patterns
            classes = [] # stores the tag of a corresponding pattern
            data_X = [] # stores all the patterns
            data_y = [] # stores the tag associated with each pattern in data_X

            for intent in data['intents']:
                for pattern in intent['patterns']:
                    tokens = nltk.word_tokenize(pattern) # tokenize the pattern
                    words.extend(tokens) # Append tokens into list
                    data_X.append(pattern)
                    data_y.append(intent['tag'])

                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

            return words, classes, data_X, data_y
        except Exception as e:
            print(f'Error cleaning data: {e}')
            raise


    def generate_bag_of_words(self, classes, data_X, data_y, words, lemmatizer):
        """
        Generates the bag of words for the training data.
        Returns the training data.
        """

        try:
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
                # Then in the output_row list, set the value at that index to 1. 
                # The rest will be 0
                output_row[classes.index(data_y[index])] = 1
                training.append([bag_of_words, output_row])

            return training
        except Exception as e:
            print(f'Error generating bag of words: {e}')
            raise


    def construct_model(self, inputSize: int, outputSize: int):
        """
        Constructs the model.
        Returns the model.
        """
        
        try:
            model = Sequential()
            model.add(Dense(128, input_shape=(inputSize, ), activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(outputSize, activation='softmax'))
            adam = tf.keras.optimizers.Adam(learning_rate = 0.01, decay=1e-6)
            model.compile(loss = 'categorical_crossentropy',
                        optimizer = adam,
                        metrics = ['accuracy']
                        )
            print(model.summary())
            return model
        except Exception as e:
            print(f'Error constructing model: {e}')
            raise


    def train(self, data: any):
        """
        Trains the model.
        Returns the model.
        """

        try:
            words, classes, data_X, data_y = self.clean_data(data)

            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
            words = sorted(set(words))
            classes = sorted(set(classes))

            pickle.dump(words, open('./data/words.pkl', 'wb'))
            pickle.dump(classes, open('./data/classes.pkl', 'wb'))

            training = self.generate_bag_of_words(classes, data_X, data_y, words, lemmatizer)

            random.shuffle(training)
            training = np.array(training, dtype=object)
            train_X = np.array(list(training[:, 0]))
            train_y = np.array(list(training[:, 1]))

            model = self.construct_model(len(train_X[0]), len(train_y[0]))
            early_stopping = EarlyStopping(monitor='val_loss', patience=80, restore_best_weights=True)
            model.fit(
                x=train_X, 
                y=train_y, 
                epochs=300, 
                verbose=2,
                validation_split=0.2,
                callbacks=[early_stopping]
            )

            loss, accuracy = model.evaluate(train_X, train_y)
            print(f'accuracy = {accuracy}\nloss = {loss}')

            return model
        except Exception as e:
            print(f'Error training model: {e}')
            raise


    def save_model(self, name: str, model: any):
        """
        Saves the model.
        """
        try:
            path = f'./models/{name}.h5'
            print(f'Saving model: {path}...')
            model.save(path)
        except Exception as e:
            print(f'Error saving model: {e}')
            raise


if __name__ == '__main__':
    try:
        model = ModelV1()
        data = Util.load_intents('./data/intents_v3.json')
        mlModel = model.train(data)
        model.save_model('modelV1', mlModel)
    except Exception as e:
        print(f'Error: {e}')