#!/usr/bin/python3

import sys
import functools
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow.keras as keras

seed = int(sys.argv[3])

FLOAT = np.float32
LABEL_COLUMN = 'kappa'
SELECT_COLUMNS = ['R','x','y','T','L','kappa']
DEFAULTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
TRAIN = 0.6
VALIDATE = 0.2
EPOCHS = 10000
SAMPLING_RANDOM_SEED = seed#4
EARLY_STOP_PATIENCE = 15
assert(TRAIN + VALIDATE <= 1.0)

def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=5, # Значение искусственно занижено для удобства восприятия.
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True, 
        **kwargs)
    return dataset

#dataset = pd.read_csv("kappa.csv", names=SELECT_COLUMNS, na_values = "?", comment='\t', sep=",", header = 0, skipinitialspace=True, dtype={'R':FLOAT, 'x':FLOAT, 'y':FLOAT, 'T':FLOAT, 'L':FLOAT, 'kappa':FLOAT}).sample(frac=1, random_state=0).reset_index(drop=True)

dataset = pd.read_csv("kappa.csv", dtype={'R':FLOAT, 'x':FLOAT, 'y':FLOAT, 'T':FLOAT, 'L':FLOAT, 'kappa':FLOAT, 'kappa_bulk':FLOAT}).sample(frac=1, random_state=SAMPLING_RANDOM_SEED).reset_index(drop=True)

del dataset['kappa_bulk']
total_len = len(dataset.index)
train_validate_len = int(total_len * (TRAIN + VALIDATE))
test_len = total_len - train_validate_len

train_dataset = dataset.head(train_validate_len)
test_dataset = dataset.tail(test_len)

train_stats = train_dataset.describe()
train_stats.pop("kappa")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('kappa')
test_labels = test_dataset.pop('kappa')

print(train_dataset['R'])
mean = train_stats['mean'].astype(FLOAT)
std = train_stats['std'].astype(FLOAT)
print(mean)
def norm(x):
  return (x - mean) / std

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model(layers, neurons_in_layer, activation = 'relu', optimizer = tf.keras.optimizers.RMSprop(0.0001)):
    model = keras.Sequential()
    #model.add(keras.layers.Input(shape=(len(train_dataset.keys()),)))
    model.add(keras.layers.Dense(neurons_in_layer, activation=activation, input_shape=(len(train_dataset.keys()),)))
    for i in range(1, layers):
        model.add(keras.layers.Dense(neurons_in_layer, activation=activation))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model

#for nn in [int(sys.argv[1])]:
#for nn in [int(sys.argv[2])]:
for nn in [6, 8, 10, 12, 14, 16, 18, 20]:
    for l in [2, 3, 4, 5]:
        #for a in [sys.argv[1]]:#'relu', 'sigmoid', 'tanh']:
        for a in ['relu', 'sigmoid', 'tanh']:
            model = build_model(l, nn, activation=a)
            name = 'model_%s_%d_%d' % (a, l, nn,)
            print(model.summary())

            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE)

            history = model.fit(
                normed_train_data, train_labels,
                epochs=EPOCHS, validation_split=VALIDATE, verbose=2,
                callbacks=[early_stop])

            loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
            with open('models.csv', 'a') as modelsf:
                modelsf.write('"%s_%d",%g,%g\n' % (name, seed, mae, mse,))
            model.save('models/%s_%d.h5' % (name,seed))
