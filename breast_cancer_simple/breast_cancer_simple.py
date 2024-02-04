import pandas as pd

forecasters = pd.read_csv('entry_breast.csv')
division = pd.read_csv('out_breast.csv')

from sklearn.model_selection import train_test_split
forecasters_training, forecasters_test, division_training, division_test = train_test_split(forecasters, division, test_size=0.25)