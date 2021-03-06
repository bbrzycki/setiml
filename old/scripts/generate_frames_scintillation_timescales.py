# imports
import numpy as np
from blimpy import read_header, Waterfall, Filterbank

import matplotlib.pyplot as plt

import sys, os, glob
sys.path.append("../../setigen")
import setigen as stg

# Set image parameters
# tsamp = 1.0
# fch1 = 6095.214842353016
# df = -1.0e-06

# fchans = 1024
# tchans = 64

tsamp = 18.253611008
fch1 = 6095.214842353016
df = -2.7939677238464355e-06

fchans = 1024
tchans = 32

fs = np.arange(fch1, fch1 + fchans*df, df)
ts = np.arange(0, tchans*tsamp, tsamp)

# Create folders
import errno
dir = 'scintillated_timescales_32_1024_v2-2'
labels = ['scintillated', 'constant', 'noise', 'choppy_rfi']
dirs = ['/datax/scratch/bbrzycki/data/%s/train/%s/' % (dir, label) for label in labels] \
        + ['/datax/scratch/bbrzycki/data/%s/validation/%s/' % (dir, label) for label in labels] 

for d in dirs:
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
                
# Generation
# Generate training and validation data!
datasets = [('train', 10000), ('validation', 500)]

for name, num in datasets:

    for i in range(num):

        output_fn = '/datax/scratch/bbrzycki/data/%s/%s/%s/%s_%04d.png' % (dir,name,'scintillated','scintillated',i)

        start_index = np.random.randint(0,fchans)
        drift_rate = np.random.uniform(-start_index*df/(tsamp*tchans),
                                       (fchans-1-start_index)*df/(tsamp*tchans))
        line_width = np.random.uniform(0.02, 0.03) ** 3
        #drift_rate = 0
        level = np.random.uniform(1,5)
        level = 5
        period = np.random.uniform(1,5)
        phase = np.random.uniform(0,period)
        sigma = np.random.uniform(0.1, 2)
        pulse_dir = 'rand'
        width = np.random.uniform(0.1, 2)
        pnum = 10
        amplitude = np.random.uniform(level*2/3, level)

        signal = stg.generate(ts,
                              fs,
                              stg.constant_path(f_start = fs[start_index], drift_rate = drift_rate),
                              stg.periodic_gaussian_t_profile(period, phase, sigma, pulse_dir, width, pnum, amplitude, level),
                              stg.gaussian_f_profile(width = line_width),
                              stg.constant_bp_profile(level = 1.0),
                              integrate = True)

        signal = stg.normalize(stg.inject_noise(signal), cols = 128, exclude = 0.2, use_median=False)

        plt.imsave(output_fn, signal)
        print('Saved %s of %s scintillated data for %s' % (i + 1, num, name))
        
    for i in range(num):

        output_fn = '/datax/scratch/bbrzycki/data/%s/%s/%s/%s_%04d.png' % (dir,name,'constant','constant',i)

        start_index = np.random.randint(0,fchans)
        drift_rate = np.random.uniform(-start_index*df/(tsamp*tchans),
                                       (fchans-1-start_index)*df/(tsamp*tchans))
        line_width = np.random.uniform(0.02, 0.05) ** 3
        #drift_rate = 0
        level = np.random.uniform(1,5)
        level = 5

        signal = stg.generate(ts,
                              fs,
                              stg.constant_path(f_start = fs[start_index], drift_rate = drift_rate),
                              stg.constant_t_profile(level = level),
                              stg.gaussian_f_profile(width = line_width),
                              stg.constant_bp_profile(level = 1.0),
                              integrate = True)

        signal = stg.normalize(stg.inject_noise(signal), cols = 128, exclude = 0.2, use_median=False)

        plt.imsave(output_fn, signal)
        print('Saved %s of %s constant data for %s' % (i + 1, num, name))
        
    for i in range(num):

        output_fn = '/datax/scratch/bbrzycki/data/%s/%s/%s/%s_%04d.png' % (dir,name,'choppy_rfi','choppy_rfi',i)

        start_index = np.random.randint(0,fchans)
        drift_rate = np.random.uniform(-start_index*df/(tsamp*tchans),
                                       (fchans-1-start_index)*df/(tsamp*tchans))

        line_width = np.random.uniform(0.02, 0.03) ** 3
        # RFI is at practically 0 drift rate
        drift_rate = 0
        level = np.random.uniform(1,5)
        level = 5
        spread = np.random.uniform(0.0002, 0.0003)

        signal = stg.generate(ts,
                              fs,
                              stg.choppy_rfi_path(f_start = fs[start_index], drift_rate = drift_rate, spread=spread, spread_type='gaussian'),
                              stg.constant_t_profile(level = level),
                              stg.gaussian_f_profile(width = line_width),
                              stg.constant_bp_profile(level = 1.0),
                              integrate = True)

        signal = stg.normalize(stg.inject_noise(signal), cols = 128, exclude = 0.2, use_median=False)

        plt.imsave(output_fn, signal)
        print('Saved %s of %s choppy rfi data for %s' % (i + 1, num, name))

    for i in range(num):

        output_fn = '/datax/scratch/bbrzycki/data/%s/%s/%s/%s_%04d.png' % (dir,name,'noise','noise',i)

        start_index = np.random.randint(0,fchans)
        drift_rate = np.random.uniform(-start_index*df/(tsamp*tchans),
                                       (fchans-1-start_index)*df/(tsamp*tchans))
        signal = stg.generate(ts,
                              fs,
                              stg.constant_path(f_start = fs[start_index], drift_rate = drift_rate),
                              stg.constant_t_profile(level = 0),
                              stg.gaussian_f_profile(width = line_width),
                              stg.constant_bp_profile(level = 1.0),
                              integrate = False)

        signal = stg.normalize(stg.inject_noise(signal), cols = 128, exclude = 0.2, use_median=False)

        plt.imsave(output_fn, signal)
        print('Saved %s of %s noise data for %s' % (i + 1, num, name))
