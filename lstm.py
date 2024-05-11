from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import glob
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16

a = []
with open("D:\Sign language detection\wordsselected.txt","r") as f:
    for i in f:
        a.append(i.replace("\n",""))
m_o = VGG16
DATA_PATH = os.path.join('MP_Data')
actions = np.array([i for i in sorted(a)])
label_map = {label:num for num, label in enumerate(actions)}
print(label_map)
print(np.shape(actions))
print(actions)
no_sequences = 50
sequence_length = 30
sequences, labels = [], []
for t,action in enumerate(actions):
    print(t+1,action)
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
print(np.shape(sequences))
print(np.shape(labels))

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
# X_train = X_train[:,:,1536:]
# X_test = X_test[:,:,1536:]
X_train = np.concatenate((X_train[:,:,:132], X_train[:,:,1536:]), axis=2)
X_test = np.concatenate((X_test[:,:,:132], X_test[:,:,1536:]), axis=2)
print(np.shape(X_train))
print(np.shape(y_train))

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
best_mod = tf.keras.callbacks.ModelCheckpoint(filepath='lstm{epoch:04d}.h6', save_best_only=True, monitor='val_categorical_accuracy', mode='max')

###model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
# model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback, best_mod], validation_data=(X_test, y_test))

model.save('lstmnew.h5')
