# basic cnn training
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
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
        Xa = np.empty((self.batch_size, *self.dim, self.n_channels))
        Xb = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.num_labels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            filename = prefix + 'data/2sig/train/%s.npy' % ID
            X1 = np.load(filename)
            X2 = np.copy(X1)
            
            X1 -= np.mean(X1)
            X1 /= np.std(X1)
            
            X2 -= np.mean(X2, axis=0)
            X2 /= np.std(X2, axis=0)
            
            Xstack = np.transpose(np.stack((X1, X2)), (1, 2, 0))
            
            # assert Xstack.shape[-1] == self.n_channels
            Xa[i] = X1.reshape((*self.dim, self.n_channels))
            Xb[i] = X2.reshape((*self.dim, self.n_channels))

            # Store class
            y[i] = self.labels[ID]
#         y = to_categorical(y)

        return [Xa, Xb], y

# dimensions of our images.
img_width, img_height = 32, 1024
                                                                 
tsamp = 1.4316557653333333

dir_name = prefix
h5_datasets = prefix + 'data/2sig/2sig.hdf5'
# validation_csv_fn = dir_name + 'train/validation_labels.csv'

##############################################################

from sklearn.model_selection import train_test_split

total_image_num = 120000
print(total_image_num)
train_num = int(total_image_num * 0.8)
validation_num = total_image_num - train_num

ids = ['%06d' % i for i in range(total_image_num)]
dbs = np.load(prefix + 'data/2sig/dbs.npy')
save_labels = np.load(prefix + 'data/2sig/labels.npy')
labels = {('%06d' % i): save_labels[i] for i in range(total_image_num)}

X_train, X_test = train_test_split(ids, test_size=0.2, random_state=42)

# Generators
partition = {'train' : [], 'validation' : []}
partition['train'] = X_train
partition['validation'] = X_test

##############################################################

batch_size = 32

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

from keras.layers import Conv2D, Conv3D, Input, Dense
from keras.models import Model

activation = 'relu'
def Residual(x, layers=32):
    conv = Conv2D(layers, (3, 3), padding='same')(x)
    residual =  keras.layers.add([x, conv])
    act = Activation(activation)(residual)
    normed = BatchNormalization()(act)
    return normed

inputs = Input(shape=(32, 1024, 1))
# 3x3 conv with 3 output channels (same as input channels)
r0 = Residual(inputs, 32)
strided0 = Conv2D(32, (3, 3), strides=2)(r0)
strided0 = Activation(activation)(strided0)
strided0 = BatchNormalization()(strided0)

r1 = Residual(strided0, 32)
strided1 = Conv2D(64, (3, 3), strides=2)(r1)
strided1 = Activation(activation)(strided1)
strided1 = BatchNormalization()(strided1)

r2 = Residual(strided1, 64)
strided2 = Conv2D(64, (3, 3), strides=2)(r2)
strided2 = Activation(activation)(strided2)
strided2 = BatchNormalization()(strided2)

inputsx = Input(shape=(32, 1024, 1))
# 3x3 conv with 3 output channels (same as input channels)
r0x = Residual(inputsx, 32)
strided0x = Conv2D(32, (3, 3), strides=2)(r0x)
strided0x = Activation(activation)(strided0x)
strided0x = BatchNormalization()(strided0x)

r1x = Residual(strided0x, 32)
strided1x = Conv2D(64, (3, 3), strides=2)(r1x)
strided1x = Activation(activation)(strided1x)
strided1x = BatchNormalization()(strided1x)

r2x = Residual(strided1x, 64)
strided2x = Conv2D(64, (3, 3), strides=2)(r2x)
strided2x = Activation(activation)(strided2x)
strided2x = BatchNormalization()(strided2x)

x = keras.layers.add([strided2, strided2x])
x = Activation(activation)(x)
x = BatchNormalization()(x)


# a layer instance is callable on a tensor, and returns a tensor
flat = Flatten()(x)
dense0 = Dense(64, activation=activation)(flat)
dense1 = Dense(4096, activation=activation)(dense0)
drop0 = Dropout(0.5)(dense1)
predictions = Dense(4, activation='linear')(drop0)

def index_diff(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred)) * 1024

model = Model(inputs=[inputs, inputsx], outputs=predictions)
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=[index_diff])

model.summary()

from keras.utils import plot_model
plot_model(model, to_file='2sig_resnet_2inputs.png', show_shapes=True)

#################################################################

model_fn = dir_name + 'models/2sig/resnet_2inputs_200epochs.h5'
history_fn = dir_name + 'models/2sig/resnet_2inputs_200epochs_history'

epochs = 1000

# Train
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
