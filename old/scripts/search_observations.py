##### Do imports and set resolution #####

import numpy as np
import json
from blimpy import read_header, Waterfall, Filterbank

import sys, os, glob
sys.path.append("../../setigen")
import setigen as stg

# tsamp = 1.0
# fch1 = 6095.214842353016
# df = -1.0e-06

tsamp = 18.253611008
fch1 = 6095.214842353016
df = -2.7939677238464355e-06

fchans = 1024
tchans = 32

fs = np.arange(fch1, fch1 + fchans*df, df)
ts = np.arange(0, tchans*tsamp, tsamp)

##### Load up model #####

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf

import matplotlib.pyplot as plt
from keras.preprocessing import image

# dimensions of our images.
img_width, img_height = 32, 1024

dir = 'scintillated_timescales_32_1024_v2-2'
VERSION = 'v2-2'

train_data_dir = '/datax/scratch/bbrzycki/data/%s/train/' % (dir)
validation_data_dir = '/datax/scratch/bbrzycki/data/%s/validation/' % (dir)
nb_train_samples = 10000*4
nb_validation_samples = 500*4
epochs = 100
batch_size = 64

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
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
model.add(Dense(4, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model_dir = '/datax/scratch/bbrzycki/models/%s/' % dir
filepath = model_dir + '%s_%s.h5' % ('scintillated_timescales_32_1024', VERSION)

model.load_weights(filepath)

##### Specify observation file #####

fn = '/datax/scratch/bbrzycki/data/blc00_guppi_58331_12383_DIAG_SGR_B2_0014.gpuspec.0000.fil'

##### Search! #####

notable_hits = {
    'choppy_rfi': [],
    'constant': [],
    'noise': [],
    'scintillated': [],
    'unclear': []
}

with open('notable_hits.json', 'w') as f:
    json.dump(notable_hits, f)

fch1 = read_header(fn)[b'fch1']
nchans = read_header(fn)[b'nchans']

notable_indices = []

for index in range(0, int(nchans / fchans)):
    f_stop = fch1 + index * fchans * df
    f_start = fch1 + (index + 1) * fchans * df
    frame = stg.get_data(Waterfall(fn, f_start=f_start, f_stop=f_stop))
    normalized = stg.normalize(frame, cols = 128, exclude = 0.2, use_median=False)
    
    # Predict each frame
    plt.imsave('temp_normalized.png', normalized)

    img = load_img('temp_normalized.png',False,target_size=(32, 1024))
    x = img_to_array(img)
    x = x / 255.
    x = np.expand_dims(x, axis=0)
    prob = model.predict_proba(x)[0]
    
    print(index)
    
    if prob[0] > 0.9:
        hit_type = 'choppy_rfi'
    elif prob[1] > 0.9:
        hit_type = 'constant'
    elif prob[2] > 0.9:
        hit_type = 'noise'
    elif prob[3] > 0.9:
        hit_type = 'scintillated'
    else:
        hit_type = 'unclear'
#         print('Index: %s, ???' % index)

    with open('notable_hits.json', 'r') as f:
        notable_hits = json.load(f)
    notable_hits[hit_type].append(index)
    with open('notable_hits.json', 'w') as f:
        json.dump(notable_hits, f)


print('Choppy RFI: %.02f%%' % (prob2[0] * 100))
print('Constant: %.02f%%' % (prob2[1] * 100))
print('Noise: %.02f%%' % (prob2[2] * 100))
print('Scintillated: %.02f%%' % (prob2[3] * 100))