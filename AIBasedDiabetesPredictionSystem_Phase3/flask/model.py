import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import pickle

data = pd.read_csv('diabetes.csv')

data_X = data.iloc[:,[1, 4, 5, 7]].values
data_Y = data.iloc[:,8].values

X = data_X
Y = data_Y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = data['Outcome'] )

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)

gnb.score(x_test, y_test)

y_preds = gnb.predict(x_test)

pickle.dump(gnb, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))


