
from keras.models import Sequential
from keras.layers.core import Dense

# load dataset
import pandas as pd
dataset = pd.read_csv("breastcancer.csv")

#We dont care about ID which is at 0 and diagnosis at 1 is our target so starting from 2:32
X = dataset.iloc[:, 2:32].values
y = dataset.iloc[:, 1].values #target

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
diagnoseEncode = LabelEncoder()
y = diagnoseEncode.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Data normalization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

NeuralNetworks = Sequential() # create model
NeuralNetworks.add(Dense(20, input_dim=30, activation='relu')) # hidden layer
NeuralNetworks.add(Dense(1, activation='sigmoid')) # output layer
NeuralNetworks.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
NeuralNetworks_fitted = NeuralNetworks.fit(X_train, y_train, epochs=100, verbose=0,
                                     initial_epoch=0)
print(NeuralNetworks.summary())
print(NeuralNetworks.evaluate(X_test, y_test))