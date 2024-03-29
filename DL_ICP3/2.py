from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Flatten

df = pd.read_csv('imdb_master.csv',encoding='latin-1')
print(df.head())
df = df[df['label']!='unsup'] #Dropping Unnecessary labelfrom dataset
sentences = df['review'].values
y = df['label'].values

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

#tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)

#tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)
maxReviewLength = max([len(s.split()) for s in sentences])
vocabularyLength = len(tokenizer.word_index)+1
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)
padded_train = pad_sequences(X_train_tokens,maxlen=maxReviewLength)
paded_test = pad_sequences(X_test_tokens,maxlen=maxReviewLength)

model = Sequential()
# Adding Embedding layer to model
model.add(Embedding(vocabularyLength, 50, input_length=maxReviewLength))
model.add(Flatten())
model.add(layers.Dense(300, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(padded_train,y_train, epochs=2, verbose=True, validation_data=(paded_test,y_test), batch_size=256)

test_loss, test_acc = model.evaluate(paded_test, y_test)
print(test_acc)

#bonus
import matplotlib.pyplot as plt

# accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
