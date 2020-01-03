# basic cnn training
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import multi_gpu_model, to_categorical

import keras
import tensorflow as tf

import numpy as np
import os, sys, errno

import csv
import codecs
import pickle
import h5py

import sys, os, glob
sys.path.append("/home/bryanb/cuckoo-cli/")
import cuckoo_cl

if len(sys.argv) > 1 and sys.argv[1] == '0':
    prefix = '/mnt_blpc1/datax/scratch/bbrzycki/training/training2/'
else:
    prefix = '/datax/scratch/bbrzycki/training/training2/'

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, num_labels=1, batch_size=32, dim=(32,1024), n_channels=1, n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.num_labels = num_labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.num_labels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            filename = '/datax/scratch/bbrzycki/training/training2/data/2sig/train/%s.npy' % ID
            X[i] = np.load(filename).reshape((*self.dim, self.n_channels))
            
            X[i] -= np.mean(X[i])
            X[i] /= np.std(X[i])

            # Store class
            y[i] = self.labels[ID]
#         y = to_categorical(y)

        return X, y

# dimensions of our images.
img_width, img_height = 32, 1024
                                                                 
tsamp = 1.4316557653333333

dir_name = '/datax/scratch/bbrzycki/training/training2/'
h5_datasets = '/datax/scratch/bbrzycki/training/training2/data/2sig/2sig.hdf5'
# validation_csv_fn = dir_name + 'train/validation_labels.csv'

batch_size = 64

# Parameters
params = {'dim': (32, 1024),
          'batch_size': batch_size,
          'n_channels': 1,
          'num_labels': 4,
          'shuffle': True}

##############################################################

from sklearn.model_selection import train_test_split

total_image_num = 80000
print(total_image_num)
train_num = int(total_image_num * 0.8)
validation_num = total_image_num - train_num

ids = ['%06d' % i for i in range(40000, 40000 + total_image_num)]
dbs = np.load(prefix + 'data/2sig/dbs.npy')
save_labels = np.load(prefix + 'data/2sig/labels.npy')
labels = {('%06d' % i): save_labels[i] for i in range(40000, 40000 + total_image_num)}

X_train, X_test = train_test_split(ids, test_size=0.2, random_state=42)

# Generators
partition = {'train' : [], 'validation' : []}
partition['train'] = X_train
partition['validation'] = X_test

##############################################################

batch_size = 64

# Parameters
params = {'dim': (32, 1024),
          'batch_size': batch_size,
          'n_channels': 1,
          'num_labels': 4,
          'shuffle': True}

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

##############################################################


# keras.backend.clear_session()

# Design model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 1024, 1)))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='linear'))

def index_diff(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred)) * 1024

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=[index_diff])

model.summary()

from keras.utils import plot_model
plot_model(model, to_file='2sig_simple_cnn.png', show_shapes=True)

#################################################################

model_fn = dir_name + 'models/2sig/simple_cnn_bright.h5'
history_fn = dir_name + 'models/2sig/simple_cnn_bright_history'

epochs = 50

history = model.fit_generator(generator=training_generator,
                    steps_per_epoch=train_num // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_num // batch_size,
                    callbacks=[keras.callbacks.ModelCheckpoint(model_fn, monitor='loss', verbose=0, save_best_only=True, mode='auto'), 
                               keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-7), 
                               keras.callbacks.EarlyStopping(monitor='loss', patience=10,verbose=0, mode='auto')])

model.save_weights(model_fn)
with open(history_fn, 'wb') as f:
    pickle.dump(history.history, f)