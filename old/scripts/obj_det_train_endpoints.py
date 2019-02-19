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
import codecs

from obj_det_classes import DataGenerator

def custom_loss(y_true, y_pred):
    ts, td, tl = y_true
    ps, pd, pl = y_pred
    
    

# dimensions of our images.
img_width, img_height = 32, 1024
epochs = 100

dir_name = '/datax/scratch/bbrzycki/data/obj_det_endpoints/'
csv_fn = dir_name + 'labels.csv'
model_fn = dir_name + 'model.h5'

total_image_num = 50000
training_num = int(total_image_num * 0.8)
validation_num = total_image_num - training_num

epochs = 100
batch_size = 64

# Parameters
params = {'dim': (32, 1024),
          'batch_size': 64,
          'n_channels': 1,
          'shuffle': True}

# Datasets
partition = {'train' : [], 'validation' : []}
labels = {}

# max_drift = 31.225e-6

with open(csv_fn, 'r') as f:
    reader = csv.reader(x.replace('\0', '') for x in f)
    for i, row in enumerate(reader):
        fn, s, e, l = row
        # label = [s, d, l]
        # print(i, d)
        label = [s, e]
#        altered = d * 1000000
#        label = [altered]
        if i < training_num:
            partition['train'].append(fn)
        else:
            partition['validation'].append(fn)
        labels[fn] = label
 
# labels = {key: (value / max_drift) for (key, value) in labels.items()}
    
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
model.add(Dropout(0.4))
model.add(Dense(2, activation='linear'))

model.compile(loss='mean_squared_error',
              optimizer='adam')

model.fit_generator(generator=training_generator,
                    steps_per_epoch=training_num // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_num // batch_size,
                    use_multiprocessing=True,
                    workers=4,
                    callbacks=[keras.callbacks.ModelCheckpoint(model_fn, monitor='loss', verbose=0, save_best_only=True, mode='auto'), keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001), keras.callbacks.EarlyStopping(monitor='loss', patience=10,verbose=0, mode='auto')])

model.save_weights(model_fn)
model.summary()
