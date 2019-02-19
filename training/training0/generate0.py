import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from blimpy import read_header, Waterfall, Filterbank

import sys, os, glob, errno
import csv

sys.path.append("../../../setigen/")
import setigen as stg

tsamp = 1.4316557653333333
fch1 = 3751.464843051508
df = -1.3969838619232178e-06

fchans = 1024
tchans = 32
fs = np.arange(fch1, fch1 + fchans*df, df)
ts = np.arange(0, tchans*tsamp, tsamp)

#################################################################

def generate_signal(sig_type='constant',
                    line_width=0.02**3,
                    level=10,
                    bias_no_drift=0,
                    **kwargs):
    # bias no drift is fraction of the time drift rate should be set to 0
    start_index = np.random.randint(0, fchans)
    if np.random.uniform(0, 1) < bias_no_drift:
        drift_rate = 0
    else:
        drift_rate = np.random.uniform(-start_index*df/(tsamp*tchans),
                                   (fchans-1-start_index)*df/(tsamp*tchans))
    # Placeholder for non rfi types, which would have spread = 0 anyway
    spread = 0
    
    if sig_type == 'noise':
        signal = stg.generate(ts,
                              fs,
                              stg.constant_path(f_start = fs[start_index], drift_rate = drift_rate),
                              stg.constant_t_profile(level = 0),
                              stg.gaussian_f_profile(width = line_width),
                              stg.constant_bp_profile(level = 1.0),
                              integrate = False)
    elif sig_type == 'constant':
        signal = stg.generate(ts,
                              fs,
                              stg.constant_path(f_start = fs[start_index], drift_rate = drift_rate),
                              stg.constant_t_profile(level = level),
                              stg.gaussian_f_profile(width = line_width),
                              stg.constant_bp_profile(level = 1.0),
                              integrate = True)
    elif sig_type == 'simple_rfi':
        spread = kwargs['spread'] # np.random.uniform(0.0002, 0.0003)

        signal = stg.generate(ts,
                              fs,
                              stg.choppy_rfi_path(f_start = fs[start_index], drift_rate = drift_rate, spread=spread, spread_type='gaussian'),
                              stg.constant_t_profile(level = level),
                              stg.gaussian_f_profile(width = line_width),
                              stg.constant_bp_profile(level = 1.0),
                              integrate = True)
    elif sig_type == 'scintillated':
        period = kwargs['period'] # np.random.uniform(1,5)
        phase = kwargs['phase'] # np.random.uniform(0, period)
        sigma = kwargs['sigma'] # np.random.uniform(0.1, 2)
        pulse_dir = kwargs['pulse_dir'] #'rand'
        width = kwargs['width'] # np.random.uniform(0.1, 2)
        pnum = kwargs['pnum'] # 10
        amplitude = kwargs['amplitude'] # np.random.uniform(level*2/3, level)

        signal = stg.generate(ts,
                              fs,
                              stg.constant_path(f_start = fs[start_index], drift_rate = drift_rate),
                              stg.periodic_gaussian_t_profile(period, phase, sigma, pulse_dir, width, pnum, amplitude, level),
                              stg.gaussian_f_profile(width = line_width),
                              stg.constant_bp_profile(level = 1.0),
                              integrate = True)
    return signal, [start_index, drift_rate, line_width, level, spread]

####################################################################

# Create folders
training_set = 0

prefix = '/datax/scratch/bbrzycki/training/training%d' % training_set
labels = ['scintillated', 'constant', 'noise', 'simple_rfi']
dirs = ['%s/train/%s/' % (prefix, label) for label in labels] \
        + ['%s/validation/%s/' % (prefix, label) for label in labels] 

for d in dirs:
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
# Make csv to save data
csv_fn = '%s/%s' % (prefix, 'labels.csv')
                
# Generate training and validation data!
datasets = [('train', 5000), ('validation', 500)]

with open(csv_fn, 'w') as f:
    writer = csv.writer(f)
    for label in labels:
        for process, num in datasets:
            for i in range(num):
                output_prefix = '%s/%s/%s/%s_%04d' % (prefix, process, label, label, i,)
                
                ##############################################################################
                # Generate signals -- these parameters should be tuned to whatever makes sense
                level = 10
                line_width = np.random.uniform(0.015, 0.03) ** 3
                
                if label == 'noise':
                    result = generate_signal(sig_type='noise')
                elif label == 'constant':
                    result = generate_signal(sig_type='constant',
                                             line_width=line_width,
                                             level=level,
                                             bias_no_drift=0.5,)
                elif label == 'simple_rfi':
                    spread = np.random.uniform(0.003, 0.017) ** 2
                    result = generate_signal(sig_type='simple_rfi',
                                             line_width=line_width,
                                             level=level,
                                             spread=spread,
                                             bias_no_drift=1,)
                elif label == 'scintillated':
                    period = np.random.uniform(1,5)
                    phase = np.random.uniform(0,period)
                    sigma = np.random.uniform(0.1, 2)
                    pulse_dir = 'rand'
                    width = np.random.uniform(0.1, 2)
                    pnum = 10
                    amplitude = np.random.uniform(level*2/3, level)

                    result = generate_signal(sig_type='scintillated',
                                             line_width=line_width,
                                             level=level,
                                             bias_no_drift=0.5,
                                             period=period,
                                             phase=phase,
                                             sigma=sigma,
                                             pulse_dir=pulse_dir,
                                             width=width,
                                             pnum=pnum,
                                             amplitude=amplitude)
                    
                signal, [start_index, drift_rate, line_width, level, spread] = result
            
                # Normalize and write data
                normalized_signal = stg.normalize(stg.inject_noise(signal), cols = 128, exclude = 0.2, use_median=False)

                plt.imsave(output_prefix + '.png', normalized_signal)
                np.save(output_prefix + '.npy', normalized_signal)
                
                writer.writerow([output_prefix, start_index, drift_rate, line_width, level, spread])
                print([output_prefix, start_index, drift_rate, line_width, level, spread])
                
                print('Saved %s of %s signal data for %s, %s' % (i + 1, num, label, process))