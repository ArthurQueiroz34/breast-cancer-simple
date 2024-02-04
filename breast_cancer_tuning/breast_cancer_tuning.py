import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

forecasters = pd.read_csv('entry_breast.csv')
division = pd.read_csv('out_breast.csv')

def createNetwork(optimizer, loos, kernel_initializer, activation, neurons):
    classifier = Sequential()
    classifier.add(Dense(units = neurons, activation = activation, 
                     kernel_initializer= kernel_initializer, input_dim = 30))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = neurons, activation = activation, 
                     kernel_initializer= kernel_initializer))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = loos, 
                   metrics = ['binary_accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = createNetwork)
parameters = {'batch_size': [10, 30],
              'epochs': [50, 100],
              'optimizer': ['adam', 'sgd'],
              'loos': ['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tahn'],
              'neurons': [16, 8]}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5)
grid_search = grid_search.fit(forecasters, division)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_