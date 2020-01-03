import matplotlib.pyplot as plt

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
import pandas as pd

import os, sys, errno

import csv
import codecs
import pickle
import h5py
import glob

from blimpy import read_header, Waterfall, Filterbank


if len(sys.argv) > 1 and sys.argv[1] == '0':
    prefix = '/mnt_blpc1/datax/scratch/bbrzycki/training/training3/'
else:
    prefix = '/datax/scratch/bbrzycki/training/training3/'

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
            filename = ID
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

dir_name = '/datax/scratch/bbrzycki/training/training3/'

##############################################################

from sklearn.model_selection import train_test_split

filenames = set(glob.glob(prefix + 'data/test/*sig*rfi*.npy'))
filenames = list(filenames - set(glob.glob(prefix + 'data/test/*sig*rfi*_128x256.npy')))
print(len(filenames))

test_num = len(filenames)

noise_labels = np.load('/datax/scratch/bbrzycki/training/training3/data/test/noise_labels.npy').item()
signal_labels = np.load('/datax/scratch/bbrzycki/training/training3/data/test/signal_labels.npy').item()
frame_param_labels = np.load('/datax/scratch/bbrzycki/training/training3/data/test/frame_param_labels.npy').item()
labels = {fn: frame_params[0] for (fn, frame_params) in frame_param_labels.items()}

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
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error',
              optimizer='adam')

model.load_weights(prefix + 'models/simple_cnn.h5')

predictions = {}
prediction_filenames = np.random.choice(filenames, 1000, replace=False)
for i, fn in enumerate(prediction_filenames):
    frame = np.load(fn)
    label = labels[fn]

    X = frame.reshape((1, 32, 1024, 1))

    X -= np.mean(X)
    X /= np.std(X)
    
    predictions[fn] = model.predict(X)[0][0]

np.save('prediction_filenames.npy', prediction_filenames)

y_true = []
ml_pred = []

for i, fn in enumerate(prediction_filenames):
    ml_pred.append(round(predictions[fn]))
    y_true.append(labels[fn])
    
np.save('y_true.npy', y_true)
np.save('ml_pred.npy', ml_pred)
    
# import edited version of turbo_seti
sys.path.insert(1,'/home/bryanb/turbo_seti/')
from turbo_seti.findoppler.findopp import FinDoppler

# This is a hack... necessary to make turbo_seti work?
obs_info = {}
obs_info['pulsar'] = 0  # Bool if pulsar detection.
obs_info['pulsar_found'] = 0  # Bool if pulsar detection.
obs_info['pulsar_dm'] = 0.0  # Pulsar expected DM.
obs_info['pulsar_snr'] = 0.0 # SNR
obs_info['pulsar_stats'] = np.zeros(6)
obs_info['RFI_level'] = 0.0
obs_info['Mean_SEFD'] = 0.0
obs_info['psrflux_Sens'] = 0.0
obs_info['SEFDs_val'] = [0.0]
obs_info['SEFDs_freq'] = [0.0]
obs_info['SEFDs_freq_up'] = [0.0]

fil = Waterfall('/home/bryanb/setigen/setigen/assets/sample.fil')

headers = ['Top_Hit_#', 
           'Drift_Rate', 
           'SNR',
           'Uncorrected_Frequency',
           'Corrected_Frequency', 
           'Index', 
           'freq_start', 
           'freq_end',
           'SEFD', 
           'SEFD_freq', 
           'Coarse_Channel_Number', 
           'Full_number_of_hits']

# BE CAREFUL WITH APPENDING TO .DAT FILES

turboseti_pred = []
for fn in prediction_filenames:
    fil.data = np.load(fn)
    save_fil_fn = '/datax/scratch/bbrzycki/turbo/training3/' + os.path.split(fn)[1][:-4] + '.fil'
    fil.write_to_fil(save_fil_fn)
    
    find_seti_event = FinDoppler(save_fil_fn, 
                                 max_drift=30.0, 
                                 snr=10, 
                                 out_dir='/datax/scratch/bbrzycki/turbo/training3/', 
                                 obs_info=obs_info)
    find_seti_event.search()

    dat_fn = save_fil_fn[:-4] + '.dat'
    df = pd.read_csv(dat_fn, delim_whitespace=True, comment='#', names=headers)
    
    turboseti_pred.append(len(df.drop_duplicates('Top_Hit_#')))
    
np.save('turboseti_pred.npy', turboseti_pred)

import sklearn.metrics
def one_off_accuracy(y_true, y_pred):
    total = 0
    for i in range(len(y_true)):
        if abs(round(y_pred[i]) - y_true[i]) <= 1:
            total += 1
    return total/len(y_true)

print('ML')
print(sklearn.metrics.accuracy_score(y_true, ml_pred))
print(one_off_accuracy(y_true, ml_pred))

print('TurboSETI')
print(sklearn.metrics.accuracy_score(y_true, turboseti_pred))
print(one_off_accuracy(y_true, turboseti_pred))