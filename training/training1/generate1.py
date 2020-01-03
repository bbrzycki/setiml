import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from blimpy import read_header, Waterfall, Filterbank

import sys, os, glob, errno
import csv
import json

sys.path.append("../../../setigen/")
import setigen as stg

tsamp = 1.4316557653333333
fch1 = 6000.464843051508
df = -1.3969838619232178e-06

fchans = 1024
tchans = 32
fs = np.arange(fch1, fch1 + fchans*df, df)
ts = np.arange(0, tchans*tsamp, tsamp)

#################################################################
def choose_from_dist(dist, shape):
    return dist[np.random.randint(0, len(dist), shape)]

def make_normal(means_dist, stds_dist, mins_dist, shape):
    means = choose_from_dist(means_dist, shape)
    stds = choose_from_dist(stds_dist, shape)
    mins = choose_from_dist(mins_dist, shape)
    means = np.maximum(means, stds)
    return means, stds, mins

def generate_frame(sig_num=True,
                   sig_db=10,
                   rfi_num=0,
                   rfi_db=25,
                   means_dist=None,
                   stds_dist=None,
                   mins_dist=None,
                    **kwargs):
    
    noise_mean, noise_std, noise_min = make_normal(means_dist, stds_dist, mins_dist, 1)
    noise_frame = np.maximum(np.random.normal(noise_mean, noise_std, [tchans, fchans]), noise_min)
    frame = noise_frame
    
    rfi_indices = []
    rfi_widths = []
    for i in range(rfi_num):
        rfi_snr = np.power(10, rfi_db / 10)
        rfi_level = noise_std * rfi_snr / np.sqrt(tchans)
        rfi_start_index = np.random.randint(0, fchans)
        rfi_indices.append(rfi_start_index)
        rfi_line_width = np.random.uniform(1e-6, 30e-6)
        rfi_widths.append(rfi_line_width)
        rfi_signal = stg.generate(ts,
                                  fs,
                                  stg.constant_path(f_start=fs[rfi_start_index], drift_rate=0),
                                  stg.constant_t_profile(level=rfi_level),
                                  stg.gaussian_f_profile(width=rfi_line_width),
                                  stg.constant_bp_profile(level=1.0))
        frame += rfi_signal

    if sig_num == 1:
        snr = np.power(10, sig_db / 10)
        level = noise_std * snr / np.sqrt(tchans)
        start_index = np.random.randint(0, fchans)
        drift_rate = np.random.uniform(-start_index*df/(tsamp*tchans),
                                       (fchans-1-start_index)*df/(tsamp*tchans))
        line_width = np.random.uniform(1e-6, 30e-6)
        signal = stg.generate(ts,
                              fs,
                              stg.constant_path(f_start=fs[start_index], drift_rate=drift_rate),
                              stg.constant_t_profile(level=level),
                              stg.gaussian_f_profile(width=line_width),
                              stg.constant_bp_profile(level=1.0))
        frame += signal
    else:
        level = 0
        start_index = -1
        drift_rate = 0
        line_width = 0
    
    return frame, [sig_num, sig_db, start_index, drift_rate, line_width, rfi_num, rfi_db, rfi_indices, rfi_widths]


if __name__ == '__main__':
    ####################################################################

    # Create folders
    training_set = '1'

    prefix = '/datax/scratch/bbrzycki/training/training%s' % training_set
    dirs = ['%s/train/' % prefix, '%s/validation/' % prefix, '%s/test/' % prefix] 

    for d in dirs:
        try:
            os.makedirs(d)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
                
    real_noise = np.load('real_noise_dists.npy')
    means_dist = real_noise[:,0]
    stds_dist = real_noise[:,1]
    mins_dist = real_noise[:,2]

    # numbers for each subcategory; # rfi (0,1,2,3), # signals (0,1), db (0,5,10,15,20)
    splits = [('train', 5000), ('validation', 500), ('test', 500)]
    
    for split in splits:
        split_name, split_num = split
        
        split_csv_fn = '%s/%s/%s_labels.csv' % (prefix, split_name, split_name)
        with open(split_csv_fn, 'w') as f:
            writer = csv.writer(f)
            for rfi_num in range(4):
                for sig_num in range(2):
                    for sig_db in [0, 5, 10, 15, 20]:
                        for i in range(split_num):
                            filename = 'frame_%drfi_%dsig_%02ddb_%04d.npy' % (rfi_num, sig_num, sig_db, i)
                            output_path =  '%s/%s/%s' % (prefix, split_name, filename)

                            ##############################################################################
                            # Generate signals
                            frame, [sig_num, sig_db, start_index, drift_rate, line_width, rfi_num, rfi_db, rfi_indices, rfi_widths] = generate_frame(sig_num=sig_num,
                                           sig_db=sig_db,
                                           rfi_num=rfi_num,
                                           rfi_db=25,
                                           means_dist=means_dist,
                                           stds_dist=stds_dist,
                                           mins_dist=mins_dist)
                            # No normalize here!

                            np.save(output_path, frame)

                            writer.writerow([output_path, sig_num, sig_db, start_index, drift_rate, line_width, rfi_num, rfi_db, rfi_indices, rfi_widths])
                            print([sig_num, sig_db, start_index, drift_rate, line_width, rfi_num, rfi_db, rfi_indices, rfi_widths])

                            print('Saved %s; %s/%s for this subsplit' % (output_path, i + 1, split_num))