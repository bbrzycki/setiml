# basic cnn training
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import keras
import tensorflow as tf

import numpy as np
import os, sys, errno

import csv

from obj_det_classes import DataGenerator

def custom_loss(y_true, y_pred):
    ts, td, tl = y_true
    ps, pd, pl = y_pred
    
    

# dimensions of our images.
img_width, img_height = 32, 1024
epochs = 100

dir_name = '/datax/scratch/bbrzycki/data/obj_det/'
csv_fn = dir_name + 'labels.csv'
model_fn = dir_name + 'model.h5'

total_image_num = 50000
training_num = int(total_image_num * 0.8)
validation_num = total_image_num - training_num

epochs = 10
batch_size = 64

# Parameters
params = {'dim': (32, 1024),
          'batch_size': 64,
          'n_channels': 1,
          'shuffle': True}

# Datasets
partition = {'train' : [], 'validation' : []}
labels = {}

with open(csv_fn, 'r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        fn, s, d, l = row
        # label = [s, d, l]
        label = [s]
        if i < training_num:
            partition['train'].append(fn)
        else:
            partition['validation'].append(fn)
        labels[fn] = label
     
# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

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
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error',
              optimizer='rmsprop')

model.fit_generator(generator=training_generator,
                    steps_per_epoch=training_num // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_num // batch_size,
                    use_multiprocessing=True,
                    workers=4,
                    callbacks=[keras.callbacks.ModelCheckpoint(model_fn, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.001), keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,verbose=0, mode='auto')])

model.save_weights(model_fn)
model.summary()
