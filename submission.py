import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load data
data_path = 'Dicoding/train.csv'
df_twitter = pd.read_csv(data_path)
df_twitter.dropna(inplace=True)

# We will focus on selected_text only
df_twitter_selected = df_twitter[['selected_text', 'sentiment']]

# since we have 3 label of sentiment, we will use one hot encoding
sentiment = pd.get_dummies(df_twitter.sentiment)
df_baru = pd.concat([df_twitter_selected, sentiment], axis=1)
df_baru = df_baru.drop(columns='sentiment')

selected_text = df_baru['selected_text'].values
label = df_baru[['negative', 'neutral', 'positive']].values


x_train, x_test, y_train, y_test = train_test_split(selected_text, label, test_size=0.2)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


# Preprocessing

OOV_TOKEN = '<OOV>'
sentence_length = 20

tokenizer = Tokenizer(oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(x_train) 
tokenizer.fit_on_texts(x_test)
 
sekuens_train = tokenizer.texts_to_sequences(x_train)
sekuens_test = tokenizer.texts_to_sequences(x_test)
 
padded_train = pad_sequences(sekuens_train, maxlen=sentence_length) 
padded_test = pad_sequences(sekuens_test, maxlen=sentence_length)

# count tokenizer
word_index = tokenizer.word_index

# Callback
# create callback (stop training if accuracy > 90%)
# Define the callback function
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') > 0.90:
            print("\nValidation accuracy is above 90%, so stopping training!")
            self.model.stop_training = True
            


# Model LSTM with CNN
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word_index) + 1, 128, input_length=sentence_length),
    tf.keras.layers.Conv1D(128, 5, activation='tanh'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy',optimizer='Nadam',metrics=['accuracy'])

# Add the callback to the fit method
history = model.fit(padded_train, y_train, epochs=10, batch_size=64,
                    validation_data=(padded_test, y_test), verbose=1,
                    callbacks=[MyCallback()])

# plot history train
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

# create info plot
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')


# plot history train
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

# create info plot
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')