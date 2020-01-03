import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from blimpy import read_header, Waterfall, Filterbank

import sys, os, glob, errno
import csv
import json
import h5py

sys.path.append("../../../setigen/")
import setigen as stg

tsamp = 1.4316557653333333
fch1 = 6000.464843051508
df = -1.3969838619232178e-06

fchans = 256
tchans = 128
fs = np.arange(fch1, fch1 + fchans*df, df)
ts = np.arange(0, tchans*tsamp, tsamp)

real_noise = np.load('real_noise_dists.npy')
means_dist = real_noise[:,0]
stds_dist = real_noise[:,1]
mins_dist = real_noise[:,2]

def choose_from_dist(dist, shape):
    return dist[np.random.randint(0, len(dist), shape)]

def make_normal(means_dist, stds_dist, mins_dist, shape):
    means = choose_from_dist(means_dist, shape)
    stds = choose_from_dist(stds_dist, shape)
    mins = choose_from_dist(mins_dist, shape)
    means = np.maximum(means, stds)
    return means, stds, mins

def generate_frame(sig_num=0,
                   max_sig_num=10,
                   rfi_frac=0.5,
                   sig_snr_range=(25, 250),
                   width_range=(5, 10),
                   means_dist=None,
                   stds_dist=None,
                   mins_dist=None,
                    **kwargs):
    
    noise_mean, noise_std, noise_min = make_normal(means_dist, stds_dist, mins_dist, 1)
    noise_frame = np.maximum(np.random.normal(noise_mean, noise_std, [tchans, fchans]), noise_min)
    frame = noise_frame

    frame_info = {
        'noise': [noise_mean[0], noise_std[0], noise_min[0]],
        'signals': [],
        'frame_params': [sig_num, rfi_frac, sig_snr_range[0], sig_snr_range[1], width_range[0], width_range[1]],
    }
    
    for i in range(sig_num):
        snr = np.random.uniform(sig_snr_range[0], sig_snr_range[1])
        level = noise_std * snr / np.sqrt(tchans)
        
        start_index = np.random.randint(0, fchans)
        if np.random.rand() > rfi_frac:
            end_index = np.random.randint(0, fchans)
        else:
            end_index = start_index
        drift_rate = (end_index - start_index) / tchans * (df / tsamp)
        
        line_width = np.random.uniform(width_range[0], width_range[1]) * np.abs(df)
        
        signal = stg.generate(ts,
                              fs,
                              stg.constant_path(f_start=fs[start_index], drift_rate=drift_rate),
                              stg.constant_t_profile(level=level),
                              stg.gaussian_f_profile(width=line_width),
                              stg.constant_bp_profile(level=1.0))
        
#         sig_info = {
#             'class': 'constant', 
#             'start_index': start_index,
#             'end_index': end_index,
#             'line_width': line_width,
#             'snr': snr,
#         }
        sig_info = np.array([start_index / fchans, end_index / fchans, (end_index - start_index) / tchans, line_width / np.abs(df), snr])
    
        frame += signal
        frame_info['signals'].append(sig_info)
    for i in range(max_sig_num - sig_num):
        frame_info['signals'].append([-1, -1, 0, 0, -1])
    
        
    return frame, frame_info

if __name__ == '__main__':
    
    # Create folders
    training_set = '3'

    prefix = '/datax/scratch/bbrzycki/training/training%s' % training_set
    dirs = ['%s/data/train/' % prefix, '%s/data/test/' % prefix] 

    for d in dirs:
        try:
            os.makedirs(d)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    
    subsplits = [('train', 2000), ('test', 200)]
    
    for j, subsplit in enumerate(subsplits):
        split_name, subsplit_num = subsplit
        
        all_noise_props = {}
        all_signal_props = {}
        all_frame_params = {}

        index = 0
        
        max_sig_num = 10
        
        for sig_num in range(max_sig_num + 1):
            # Probabilities from 0 to 1 inclusive, increments of 0.1
            for rfi_frac in np.linspace(0, 1, 11):
                for i in range(subsplit_num):
                    frame, frame_info = generate_frame(sig_num=sig_num,
                           max_sig_num=max_sig_num,
                           rfi_frac=rfi_frac,
                           sig_snr_range=(25, 250),
                           width_range=(5, 10),
                           means_dist=means_dist,
                           stds_dist=stds_dist,
                           mins_dist=mins_dist)     
                    
                    # Try more descriptive naming
                    frame_fn = '%s/data/%s/%02dsig_%.01frfi_%06d_128x256.npy' % (prefix, split_name, sig_num, rfi_frac, i)
                    np.save(frame_fn, frame)
                    # identifier = [sig_num, rfi_frac, i]

                    all_noise_props[frame_fn] = (frame_info['noise'])
                    all_signal_props[frame_fn] = (frame_info['signals'])
                    all_frame_params[frame_fn] = (frame_info['frame_params'])
                    index += 1

                    print('Saved: %02d, %.01f, %06d, %s' % (sig_num, rfi_frac, i, split_name))
                    print('Size = %d, %d, %d' % (len(all_noise_props[frame_fn]), len(all_signal_props[frame_fn]), len(all_frame_params[frame_fn])))
                    print(all_noise_props[frame_fn])
                    print(all_signal_props[frame_fn])
                    print(all_frame_params[frame_fn])        

        np.save('%s/data/%s/noise_labels_128x256.npy' % (prefix, split_name), all_noise_props)
        np.save('%s/data/%s/signal_labels_128x256.npy' % (prefix, split_name), all_signal_props)
        np.save('%s/data/%s/frame_param_labels_128x256.npy' % (prefix, split_name), all_frame_params)
        
        
