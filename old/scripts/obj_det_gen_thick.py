# imports
import numpy as np
from blimpy import read_header, Waterfall, Filterbank

import sys, os, glob, errno
sys.path.append("../../setigen")
import setigen as stg

import csv

tsamp = 1.4316557653333333
fch1 = 3751.464843051508
df = -1.3969838619232178e-06

fchans = 1024
tchans = 32

fs = np.arange(fch1, fch1 + fchans*df, df)
ts = np.arange(0, tchans*tsamp, tsamp)

dir_name = '/datax/scratch/bbrzycki/data/obj_det_thick/'
try:
    os.makedirs(dir_name)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
csv_fn = dir_name + 'labels.csv'
        
total_image_num = 50000
with open(csv_fn, 'w') as f:
    writer = csv.writer(f)
    for i in range(0, 50000):
        output_fn = dir_name + '%06d.npy' % i

        start_index = np.random.randint(0,fchans)
        drift_rate = np.random.uniform(-start_index*df/(tsamp*tchans), (fchans-1-start_index)*df/(tsamp*tchans))
        line_width = np.random.uniform(0.04, 0.05) ** 3
        level = 10

        signal = stg.generate(ts,
                              fs,
                              stg.constant_path(f_start = fs[start_index], drift_rate = drift_rate),
                              stg.constant_t_profile(level = level),
                              stg.gaussian_f_profile(width = line_width),
                              stg.constant_bp_profile(level = 1.0),
                              integrate = True)

        signal = stg.normalize(stg.inject_noise(signal), cols = 128, exclude = 0.2, use_median=False)

        np.save(output_fn, signal)
        writer.writerow([output_fn, start_index, drift_rate, line_width])
        print('%06d saved' % i)
        
