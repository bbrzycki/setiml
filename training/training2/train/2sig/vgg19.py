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

            # Set up input arrays
            filename = prefix + 'data/2sig/train/%s.npy' % ID

            X1 = np.load(filename)
            X2 = np.copy(X1)
            X3 = np.copy(X1)

            X1 -= np.mean(X1, keepdims=True)
            X1 /= np.std(X1, keepdims=True)

            X2 -= np.mean(X2, axis=0, keepdims=True)
            X2 /= np.std(X2, axis=0, keepdims=True)

            X3 -= np.mean(X3, axis=1, keepdims=True)
            X3 /= np.std(X3, axis=1, keepdims=True)


            Xstack = np.transpose(np.stack((X1, X1, X1)), (1, 2, 0))

            assert Xstack.shape[-1] == self.n_channels
            X[i] = np.repeat(Xstack, repeats=2, axis=0)

            # Store class
            y[i] = self.labels[ID]
#         y = to_categorical(y)

        return X, y

# dimensions of our images.
img_width, img_height = 32*2, 1024

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

batch_size = 16

# Parameters
params = {'dim': (img_width, img_height),
          'batch_size': batch_size,
          'n_channels': 3,
          'num_labels': 4,
          'shuffle': True}

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# create the base pre-trained model
base_model = keras.applications.vgg19.VGG19(input_shape=(img_width, img_height, 3),
                        weights='imagenet',
                        include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a regression layer
predictions = Dense(4, activation='linear')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

for i, layer in enumerate(base_model.layers):
    print(i, layer.name)
    

# Filename details
model_fn = dir_name + 'models/2sig/vgg19.h5'
history_fn = dir_name + 'models/2sig/vgg19_history'

epochs = 100
    
def index_diff(y_true, y_pred):
    return K.mean((y_true - y_pred)**2)**0.5 * 1024

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
    
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam',
              loss='mse',
              metrics=[index_diff])   

# Train top layer
history = model.fit_generator(generator=training_generator,
                    steps_per_epoch=train_num // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_num // batch_size,
                    callbacks=[keras.callbacks.ModelCheckpoint(model_fn, monitor='loss', verbose=0, save_best_only=True, mode='auto'),
                               keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-7),
                               keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')])

model.save_weights(model_fn)

all_history = {}
all_history['loss'] = []
all_history['val_loss'] = []

all_history['loss'].append(history.history['loss'])
all_history['val_loss'].append(history.history['val_loss'])
with open(history_fn, 'wb') as f:
    pickle.dump(all_history, f)

# Next leg of training    
for layer in model.layers[:17]:
   layer.trainable = False
for layer in model.layers[17:]:
   layer.trainable = True

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='mse',
              metrics=[index_diff])

history = model.fit_generator(generator=training_generator,
                    steps_per_epoch=train_num // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_num // batch_size,
                    callbacks=[keras.callbacks.ModelCheckpoint(model_fn, monitor='loss', verbose=0, save_best_only=True, mode='auto'),
                               keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-7),
                               keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')])

model.save_weights(model_fn)

all_history['loss'].append(history.history['loss'])
all_history['val_loss'].append(history.history['val_loss'])
with open(history_fn, 'wb') as f:
    pickle.dump(all_history, f)

# Next leg of training    
for layer in model.layers[:12]:
   layer.trainable = False
for layer in model.layers[12:]:
   layer.trainable = True

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='mse',
              metrics=[index_diff])

history = model.fit_generator(generator=training_generator,
                    steps_per_epoch=train_num // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_num // batch_size,
                    callbacks=[keras.callbacks.ModelCheckpoint(model_fn, monitor='loss', verbose=0, save_best_only=True, mode='auto'),
                               keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-7),
                               keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')])

model.save_weights(model_fn)

all_history['loss'].append(history.history['loss'])
all_history['val_loss'].append(history.history['val_loss'])
with open(history_fn, 'wb') as f:
    pickle.dump(all_history, f)

# Next leg of training    
for layer in model.layers[:7]:
   layer.trainable = False
for layer in model.layers[7:]:
   layer.trainable = True

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='mse',
              metrics=[index_diff])

history = model.fit_generator(generator=training_generator,
                    steps_per_epoch=train_num // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_num // batch_size,
                    callbacks=[keras.callbacks.ModelCheckpoint(model_fn, monitor='loss', verbose=0, save_best_only=True, mode='auto'),
                               keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-7),
                               keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')])

model.save_weights(model_fn)

all_history['loss'].append(history.history['loss'])
all_history['val_loss'].append(history.history['val_loss'])
with open(history_fn, 'wb') as f:
    pickle.dump(all_history, f)

# Next leg of training    
for layer in model.layers[:4]:
   layer.trainable = False
for layer in model.layers[4:]:
   layer.trainable = True

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='mse',
              metrics=[index_diff])

history = model.fit_generator(generator=training_generator,
                    steps_per_epoch=train_num // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_num // batch_size,
                    callbacks=[keras.callbacks.ModelCheckpoint(model_fn, monitor='loss', verbose=0, save_best_only=True, mode='auto'),
                               keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-7),
                               keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')])

model.save_weights(model_fn)

all_history['loss'].append(history.history['loss'])
all_history['val_loss'].append(history.history['val_loss'])
with open(history_fn, 'wb') as f:
    pickle.dump(all_history, f)

# Next leg of training    
for layer in model.layers[:1]:
   layer.trainable = False
for layer in model.layers[1:]:
   layer.trainable = True

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='mse',
              metrics=[index_diff])

history = model.fit_generator(generator=training_generator,
                    steps_per_epoch=train_num // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_num // batch_size,
                    callbacks=[keras.callbacks.ModelCheckpoint(model_fn, monitor='loss', verbose=0, save_best_only=True, mode='auto'),
                               keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-7),
                               keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')])

model.save_weights(model_fn)

all_history['loss'].append(history.history['loss'])
all_history['val_loss'].append(history.history['val_loss'])
with open(history_fn, 'wb') as f:
    pickle.dump(all_history, f)

