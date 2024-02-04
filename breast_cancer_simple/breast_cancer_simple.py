import pandas as pd

forecasters = pd.read_csv('entry_breast.csv')
division = pd.read_csv('out_breast.csv')

from sklearn.model_selection import train_test_split
forecasters_training, forecasters_test, division_training, division_test = train_test_split(forecasters, division, test_size=0.25)

import keras
from keras.models import Sequential 
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = 16, activation = 'relu', 
                     kernel_initializer= 'random_uniform', input_dim = 30))
classifier.add(Dense(units = 16, activation = 'relu', 
                     kernel_initializer= 'random_uniform'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

optimizer = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', 
                   metrics = ['binary_accuracy'])

#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
#                   metrics = ['binary_accuracy'])
classifier.fit(forecasters_training, division_training,
               batch_size = 10, epochs = 100)

forecasters = classifier.predict(forecasters_test)
forecasters = (forecasters > 0.5)
from sklearn.metrics import confusion_matrix, accuracy_score
accuracy = accuracy_score(division_test, forecasters)
matrix = confusion_matrix(division_test, forecasters)

result = classifier.evaluate(forecasters_test, division_test)
