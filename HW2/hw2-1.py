import numpy as np
from sklearn.metrics import mean_squared_error
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# Input dataset for Training
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

X1_test = np.array([[0, 1, 1]])

alpha = 0.4
epochs = 25
# define the keras model
model = Sequential()
model.add(Dense(4, input_dim=3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
opt = SGD(lr=alpha)
model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['binary_accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=epochs, batch_size=1)
# make class predictions with the model
print("******TESTING******")
history = model.predict(X1_test)
print (history)



