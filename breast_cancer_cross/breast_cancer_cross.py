import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

forecasters = pd.read_csv('entry_breast.csv')
division = pd.read_csv('out_breast.csv')

def createNetwork():
    classifier = Sequential()
    classifier.add(Dense(units = 16, activation = 'relu', 
                     kernel_initializer= 'random_uniform', input_dim = 30))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 16, activation = 'relu', 
                     kernel_initializer= 'random_uniform'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    
    optimizer = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', 
                   metrics = ['binary_accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = createNetwork,
                             epochs = 100,
                             batch_size = 10)
results = cross_val_score(estimator = classifier,
                          x = forecasters, y = division,
                          cv = 10, scoring = 'accuracy')
average = results.mean()
bias = results.std()