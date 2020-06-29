from tensorflow.keras.datasets import imdb
from tensorflow.keras import models
from tensorflow.keras import layers
import datetime, os
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequences in enumerate(sequences):        
        results[i, sequences] = 1.    
    return results

def test_NN():
    



(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
train_data[0] 

# words are indexed by overall frequency in the dataset, 
# so that for instance the integer "3" encodes the 3rd most frequent word in the data. 

word_index = imdb.get_word_index()

# to see the index of Word giraffe
# to see the most frequently used word in the database
topkeyWord = [topkey for topkey, value in word_index.items() if value == 1]

print('the most frequently used word in the database {}'.format(topkeyWord))

# pre-process the training data with one hot encoding
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


# convert binary labels into float
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# out of training seperate validation data 
x_val = x_train[:10000]
partial_x_train = x_train[10000: ]

# out of training seperate validation labels
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# build a NN model framework (a computational graph)
model = models.Sequential()
model.add(layers.Dense(16, activation= 'relu', input_shape= (10000, )))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Configures the model for training
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Trains the model for a fixed number of epochs (iterations on a dataset)
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=10,
                    batch_size=512,
                    validation_data=(x_val, y_val)) 

# Returns the loss value & metrics values for the model in test mode.
results = model.evaluate(x_test, y_test)
print('results {}'.format(results))

# Generates output predictions for the input samples x_test
print('predicted test results {}'.format(model.predict(x_test)))

import matplotlib.pyplot as plt   

history_dict = history.history
history_dict.keys()


plt.plot(history_dict['val_acc'], label='Validation Accuracy', color ='blue',marker = '*')
plt.plot(history_dict['acc'], label='Training Accuracy', color='red',marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.plot(history_dict['val_loss'], label='Validation Loss', color ='blue',marker = '*')
plt.plot(history_dict['loss'], label='Training Loss', color='red',marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()