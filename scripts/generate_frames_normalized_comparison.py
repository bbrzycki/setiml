# imports
import numpy as np
from blimpy import read_header, Waterfall, Filterbank

import matplotlib.pyplot as plt

import sys, os, glob
sys.path.append("../../setigen")
import setigen as stg

# Set image parameters
tsamp = 18.253611008
fch1 = 6095.214842353016
df = -2.7939677238464355e-06

fchans = 1024
tchans = 16

fs = np.arange(fch1, fch1 + fchans*df, df)
ts = np.arange(0, tchans*tsamp, tsamp)

# Create folders
import errno
dir = 'normalized_comparison'
sets = ['raw', 'normalized', 'normalized_excluded']
for set in sets:
    # labels = ['pulsed_nonzero', 'pulsed_zero', 'constant_nonzero', 'constant_zero', 'noise']
    labels = ['pulsed', 'constant', 'noise']
    dirs = ['/datax/scratch/bbrzycki/data/%s/%s/train/%s/' % (dir, set, label) for label in labels] \
            + ['/datax/scratch/bbrzycki/data/%s/%s/validation/%s/' % (dir, set, label) for label in labels] 

    for d in dirs:
        try:
            os.makedirs(d)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
                
# Generation
# Generate training and validation data!
datasets = [('train', 5000), ('validation', 500)]

for set in sets:
    for name, num in datasets:

        for i in range(num):

            output_fn = '/datax/scratch/bbrzycki/data/%s/%s/%s/%s/%s_%04d.png' % (dir,set,name,'pulsed','pulsed',i)

            start_index = np.random.randint(0,fchans)
            drift_rate = np.random.uniform(-start_index*df/(tsamp*tchans),
                                           (fchans-1-start_index)*df/(tsamp*tchans))
            #drift_rate = 0
            level = np.random.uniform(2, 5)
            width = np.random.uniform(0.02, 0.05) ** 3
            
            amplitude = np.random.uniform(level/4, level)
            # amplitude = level
            period = np.random.uniform(50, 100)
            # width = np.random.uniform(0.000009, 0.000225)
            

            signal = stg.generate(ts,
                                      fs,
                                      stg.constant_path(f_start = fs[start_index], drift_rate = drift_rate),
                                      stg.sine_t_profile(period = period,
                                                         phase = 0,
                                                         amplitude = amplitude,
                                                         level = level),
                                      stg.gaussian_f_profile(width = width),
                                      stg.constant_bp_profile(level = 1.0))

            if set == 'raw':
                signal = stg.inject_noise(signal)
            elif set == 'normalized':
                signal = stg.normalize(stg.inject_noise(signal), cols = 0, exclude = 0.0, use_median=False)
            elif set == 'normalized_excluded':
                signal = stg.normalize(stg.inject_noise(signal), cols = 1, exclude = 0.2, use_median=False)

            plt.imsave(output_fn, signal)
            print('Saved %s of %s pulsed data for %s, set %s' % (i, num, name, set))

        for i in range(num):

            output_fn = '/datax/scratch/bbrzycki/data/%s/%s/%s/%s/%s_%04d.png' % (dir,set,name,'constant','constant',i)

            start_index = np.random.randint(0,fchans)
            drift_rate = np.random.uniform(-start_index*df/(tsamp*tchans),
                                           (fchans-1-start_index)*df/(tsamp*tchans))
            level = np.random.uniform(2, 5)
            width = np.random.uniform(0.02, 0.05) ** 3

            signal = stg.generate(ts,
                                      fs,
                                      stg.constant_path(f_start = fs[start_index], drift_rate = drift_rate),
                                      stg.constant_t_profile(level = level),
                                      stg.gaussian_f_profile(width = width),
                                      stg.constant_bp_profile(level = 1.0))

            if set == 'raw':
                signal = stg.inject_noise(signal)
            elif set == 'normalized':
                signal = stg.normalize(stg.inject_noise(signal), cols = 0, exclude = 0.0, use_median=False)
            elif set == 'normalized_excluded':
                signal = stg.normalize(stg.inject_noise(signal), cols = 1, exclude = 0.2, use_median=False)

            plt.imsave(output_fn, signal)
            print('Saved %s of %s constant data for %s, set %s' % (i, num, name, set))

        for i in range(num):

            output_fn = '/datax/scratch/bbrzycki/data/%s/%s/%s/%s/%s_%04d.png' % (dir,set,name,'noise','noise',i)

            # level = 0 for no signal
            signal = stg.generate(ts,
                                      fs,
                                      stg.constant_path(f_start = fs[0], drift_rate = 0),
                                      stg.constant_t_profile(level = 0),
                                      stg.gaussian_f_profile(width = 0.00002),
                                      stg.constant_bp_profile(level = 1.0))

            if set == 'raw':
                signal = stg.inject_noise(signal)
            elif set == 'normalized':
                signal = stg.normalize(stg.inject_noise(signal), cols = 0, exclude = 0.0, use_median=False)
            elif set == 'normalized_excluded':
                signal = stg.normalize(stg.inject_noise(signal), cols = 1, exclude = 0.2, use_median=False)

            plt.imsave(output_fn, signal)
            print('Saved %s of %s noise data for %s, set %s' % (i, num, name, set))