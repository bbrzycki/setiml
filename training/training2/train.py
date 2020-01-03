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

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset_fn, list_IDs, labels, num_labels=1, batch_size=32, dim=(32,1024), n_channels=1, n_classes=10, shuffle=True):
        'Initialization'
        self.dataset_fn = dataset_fn
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
            with h5py.File(self.dataset_fn, 'r') as f:
                X[i] = np.array(f['train'][ID]['frame']).reshape((*self.dim, self.n_channels))
            # X[i] = np.load(ID).reshape((*self.dim, self.n_channels))
#             X[i] /= np.max(X[i])
            X[i] -= np.mean(X[i])
            X[i] /= np.std(X[i])

            # Store class
            y[i] = self.labels[ID]
#         y = to_categorical(y)

        return X, y


def custom_loss(y_true, y_pred):
    ts, td, tl = y_true
    ps, pd, pl = y_pred
                                                                 
cuckoo_clock = cuckoo_cl.Clock()

# dimensions of our images.
img_width, img_height = 32, 1024
                                                                 
tsamp = 1.4316557653333333

dir_name = '/datax/scratch/bbrzycki/training/training2/'
h5_datasets = dir_name + 'datasets.hdf5'
# validation_csv_fn = dir_name + 'train/validation_labels.csv'
model_fn = dir_name + 'models/model.h5'
history_fn = dir_name + 'models/model_history'

epochs = 100
batch_size = 64

# Parameters
params = {'dim': (32, 1024),
          'batch_size': batch_size,
          'n_channels': 1,
          'num_labels': 3,
          'shuffle': True}

# Datasets
partition = {'train' : [], 'validation' : []}
labels = {}

# max_drift = 31.225e-6
relevant_ids = []
with h5py.File(h5_datasets, 'r') as f:
    for i in range(150000, 200000):
        ID = '%06d' % i
        print('On %s' % ID)
        
        class_nums = f['train'][ID].attrs['class_nums']
        sig_db = f['train'][ID].attrs['sig_db']
                                                                  
        if np.all(class_nums == [1, 0]):
            relevant_ids.append(ID)
            
            start_index, end_index, line_width, snr, class_label = np.squeeze(f['train'][ID]['signals_info'])
            # All between 0 and 1
            label = (start_index / 1024, end_index / 1024, line_width / (30e-6))
            labels[ID] = label

total_image_num = len(relevant_ids)
print(total_image_num)
train_num = int(total_image_num * 0.8)
validation_num = total_image_num - train_num

partition['train'] = relevant_ids[:train_num]
partition['validation'] = relevant_ids[train_num:]
        
 
# labels = {key: (value / max_drift) for (key, value) in labels.items()}

# Generators
training_generator = DataGenerator(h5_datasets, partition['train'], labels, **params)
validation_generator = DataGenerator(h5_datasets, partition['validation'], labels, **params)

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
model.add(Dense(3, activation='linear'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit_generator(generator=training_generator,
                    steps_per_epoch=train_num // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_num // batch_size,
                    callbacks=[keras.callbacks.ModelCheckpoint(model_fn, monitor='loss', verbose=0, save_best_only=True, mode='auto'), keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001), keras.callbacks.EarlyStopping(monitor='loss', patience=10,verbose=0, mode='auto')])

model.save_weights(model_fn)
with open(history_fn, 'wb') as f:
    pickle.dump(history.history, f)

model.summary()

cuckoo_clock.send_email('Training finished!', 'Training on digilab finished')