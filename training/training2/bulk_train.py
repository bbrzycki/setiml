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
            filename = '/datax/scratch/bbrzycki/training/training2/data/1sig/train/%s.npy' % ID
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
h5_datasets = '/datax/scratch/bbrzycki/training/training2/data/1sig/1sig.hdf5'
# validation_csv_fn = dir_name + 'train/validation_labels.csv'

batch_size = 64

# Parameters
params = {'dim': (32, 1024),
          'batch_size': batch_size,
          'n_channels': 1,
          'num_labels': 2,
          'shuffle': True}


##############################################################

total_image_num = 120000
print(total_image_num)
train_num = int(total_image_num * 0.8)
validation_num = total_image_num - train_num

ids = ['%06d' % i for i in range(total_image_num)]
dbs = np.load('/datax/scratch/bbrzycki/training/training2/data/1sig/dbs.npy')
save_labels = np.load('/datax/scratch/bbrzycki/training/training2/data/1sig/labels.npy')
labels = {('%06d' % i): save_labels[i] for i in range(total_image_num)}

# Generators
partition = {'train' : [], 'validation' : []}
partition['train'] = ids[:train_num]
partition['validation'] = ids[train_num:]

##############################################################

batch_size = 64

# Parameters
params = {'dim': (32, 1024),
          'batch_size': batch_size,
          'n_channels': 1,
          'num_labels': 2,
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
model.add(Dense(2, activation='linear'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

#################################################################

model_fn = dir_name + 'models/1sig/simple_cnn.h5'
history_fn = dir_name + 'models/1sig/simple_cnn_history'

epochs = 50

history = model.fit_generator(generator=training_generator,
                    steps_per_epoch=train_num // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_num // batch_size,
                    callbacks=[keras.callbacks.ModelCheckpoint(model_fn, monitor='loss', verbose=0, save_best_only=True, mode='auto'), 
                               keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001), 
                               keras.callbacks.EarlyStopping(monitor='loss', patience=10,verbose=0, mode='auto')])

model.save_weights(model_fn)
with open(history_fn, 'wb') as f:
    pickle.dump(history.history, f)

#################################################################

from keras.layers import Conv2D, Input, Dense
from keras.models import Model

def Residual(x, layers=32):
    conv = Conv2D(layers, (3, 3), padding='same')(x)
    return keras.layers.add([x, conv])

inputs = Input(shape=(32, 1024, 1))
# 3x3 conv with 3 output channels (same as input channels)
r0 = Residual(inputs, 32)
strided0 = Conv2D(32, (3, 3), strides=2)(r0)

r1 = Residual(strided0, 32)
strided1 = Conv2D(64, (3, 3), strides=2)(r1)

r2 = Residual(strided1, 64)
strided2 = Conv2D(64, (3, 3), strides=2)(r2)

# a layer instance is callable on a tensor, and returns a tensor
flat = Flatten()(strided2)
dense0 = Dense(64, activation='relu')(flat)
dense1 = Dense(64, activation='relu')(dense0)
drop0 = Dropout(0.5)(dense1)
predictions = Dense(2, activation='linear')(drop0)

model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)

#################################################################

model_fn = dir_name + 'models/1sig/resnet.h5'
history_fn = dir_name + 'models/1sig/resnet_history'

epochs = 50

# Train
history = model.fit_generator(generator=training_generator,
                    steps_per_epoch=train_num // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_num // batch_size,
                    callbacks=[keras.callbacks.ModelCheckpoint(model_fn, monitor='loss', verbose=0, save_best_only=True, mode='auto'), 
                               keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001), 
                               keras.callbacks.EarlyStopping(monitor='loss', patience=10,verbose=0, mode='auto')])

model.save_weights(model_fn)
with open(history_fn, 'wb') as f:
    pickle.dump(history.history, f)

#################################################################

from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Conv2D, Input, Dense
from keras.models import Model

def Residual(x, layers=32):
    conv = Conv2D(layers, (3, 3), padding='same')(x)
    return keras.layers.add([x, conv])

def Inception(x, layers=64):
    tower_1 = Conv2D(layers, (1, 1), padding='same', activation='relu')(x)
    tower_1 = Conv2D(layers, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(layers, (1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(layers, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_3 = Conv2D(layers, (1, 1), padding='same', activation='relu')(tower_3)

    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
    return output

inputs = Input(shape=(32, 1024, 1))
# 3x3 conv with 3 output channels (same as input channels)
i0 = Inception(inputs, 32)
strided0 = Conv2D(32, (5, 3), strides=(1, 2))(i0)
c1 = Conv2D(32, (3, 3), strides=2)(strided0)

# r1 = Residual(strided0, 32)
# strided1 = Conv2D(64, (3, 3), strides=2)(r1)

# r2 = Residual(strided1, 64)
# strided2 = Conv2D(64, (3, 3), strides=2)(r2)

# a layer instance is callable on a tensor, and returns a tensor
flat = Flatten()(c1)
dense0 = Dense(32, activation='relu')(flat)
dense1 = Dense(64, activation='relu')(dense0)
drop0 = Dropout(0.5)(dense1)
predictions = Dense(2, activation='linear')(drop0)

model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

from keras.utils import plot_model
plot_model(model, to_file='inception_model.png', show_shapes=False)

#################################################################

model_fn = dir_name + 'models/1sig/inception.h5'
history_fn = dir_name + 'models/1sig/inception_history'

epochs = 50

# Train
history = model.fit_generator(generator=training_generator,
                    steps_per_epoch=train_num // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_num // batch_size,
                    callbacks=[keras.callbacks.ModelCheckpoint(model_fn, monitor='loss', verbose=0, save_best_only=True, mode='auto'), 
                               keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001), 
                               keras.callbacks.EarlyStopping(monitor='loss', patience=10,verbose=0, mode='auto')])

model.save_weights(model_fn)
with open(history_fn, 'wb') as f:
    pickle.dump(history.history, f)

