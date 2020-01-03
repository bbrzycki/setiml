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

def generate_frame(sig_num=0,
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
    
    frame_info = {
        'noise_mean': noise_mean,
        'noise_std': noise_std,
        'noise_min': noise_min,
        'signals': [],
    }
    
    rfi_indices = []
    rfi_widths = []
    for i in range(rfi_num):
        rfi_snr = np.power(10, rfi_db / 10)
        rfi_level = noise_std * rfi_snr / np.sqrt(tchans)
        rfi_start_index = np.random.randint(0, fchans)
        rfi_end_index = rfi_start_index
        rfi_drift_rate = 0
        rfi_indices.append(rfi_start_index)
        rfi_line_width = np.random.uniform(1e-6, 30e-6)
        rfi_widths.append(rfi_line_width)
        rfi_signal = stg.generate(ts,
                                  fs,
                                  stg.constant_path(f_start=fs[rfi_start_index], drift_rate=rfi_drift_rate),
                                  stg.constant_t_profile(level=rfi_level),
                                  stg.gaussian_f_profile(width=rfi_line_width),
                                  stg.constant_bp_profile(level=1.0))
        
        sig_info = {
            'class': 'rfi', 
            'start_index': rfi_start_index,
            'end_index': rfi_end_index,
            'line_width': rfi_line_width,
            'snr': rfi_snr,
        }
        sig_info = np.array([rfi_start_index, rfi_end_index, rfi_line_width, rfi_snr, 1])
        frame += rfi_signal
        frame_info['signals'].append(sig_info)

    for i in range(sig_num):
        snr = np.power(10, sig_db / 10)
        level = noise_std * snr / np.sqrt(tchans)
        start_index = np.random.randint(0, fchans)
        end_index = np.random.randint(0, fchans)
        drift_rate = (end_index - start_index) * df / (tsamp * tchans)
        # drift_rate = np.random.uniform(-start_index*df/(tsamp*tchans),
        #                                (fchans-1-start_index)*df/(tsamp*tchans))
        line_width = np.random.uniform(1e-6, 30e-6)
        signal = stg.generate(ts,
                              fs,
                              stg.constant_path(f_start=fs[start_index], drift_rate=drift_rate),
                              stg.constant_t_profile(level=level),
                              stg.gaussian_f_profile(width=line_width),
                              stg.constant_bp_profile(level=1.0))
        sig_info = {
            'class': 'constant', 
            'start_index': start_index,
            'end_index': end_index,
            'line_width': line_width,
            'snr': snr,
        }
        sig_info = np.array([start_index, end_index, line_width, snr, 0])
        frame += signal
        frame_info['signals'].append(sig_info)
    else:
        level = 0
        start_index = -1
        drift_rate = 0
        line_width = 0
    
    return frame, frame_info


if __name__ == '__main__':
    ####################################################################

    # Create folders
    training_set = '2'

    prefix = '/datax/scratch/bbrzycki/training/training%s' % training_set
    # dirs = ['%s/train/' % prefix, '%s/validation/' % prefix, '%s/test/' % prefix] 

    # for d in dirs:
    #     try:
    #         os.makedirs(d)
    #     except OSError as e:
    #         if e.errno != errno.EEXIST:
    #             raise
                
    real_noise = np.load('real_noise_dists.npy')
    means_dist = real_noise[:,0]
    stds_dist = real_noise[:,1]
    mins_dist = real_noise[:,2]

    # numbers for each subcategory; # rfi (0,1,2,3), # signals (0,1), db (0,5,10,15,20)
    splits = [('train', 20000), ('test', 4000)]
    # update_splits = [('train', 40000), ('test', 8000)]
    
    for rfi_num in range(2):
        sig_num = 1
        set_dir_name = '%dsig' % (1 + rfi_num)
        with h5py.File('%s/data/%s/%s.hdf5' % (prefix, set_dir_name, set_dir_name), 'w') as f:
            f_metadata = {
                'tsamp': tsamp,
                'fch1': fch1,
                'df': df,
                'fchans': fchans,
                'tchans': tchans,
                'classes': ['constant', 'rfi'],
            }
            f.attrs.update(f_metadata)

            for j, split in enumerate(splits):
                split_name, split_num = split

                g = f.create_group(split_name)
                g_metadata = {
                    'sample_num': split_num,
                }
                g.attrs.update(g_metadata)

                index = 0
                for sig_db in [0, 5, 10, 15, 20, 25]:
                    for i in range(split_num):
                        true_index = index # + splits[j][1] * 20
                        frame_group = g.create_group('%06d' % true_index)

                        ##############################################################################
                        # Generate signals
                        frame, frame_info = generate_frame(sig_num=sig_num,
                                       sig_db=sig_db,
                                       rfi_num=rfi_num,
                                       rfi_db=25,
                                       means_dist=means_dist,
                                       stds_dist=stds_dist,
                                       mins_dist=mins_dist)
                        # No normalize here!

                        frame_metadata = {
                            'noise_mean': frame_info['noise_mean'],
                            'noise_std': frame_info['noise_std'],
                            'noise_min': frame_info['noise_min'],
                            'class_nums': [sig_num, rfi_num],
                            'sig_db': sig_db,
                        }
                        frame_group.attrs.update(frame_metadata)

                        # frame = frame_group.create_dataset('frame', data=frame)
                        signals_info = frame_group.create_dataset('signals_info', data=np.array(frame_info['signals']))

                        np.save('%s/data/%s/%s/%06d.npy' % (prefix, set_dir_name, split_name, true_index), frame)

                        print('Saved %s, %s, %s; %s/%s for this subsplit, %06d of %06d total (%06d)' % (split_name, rfi_num, sig_db, i + 1, split_num, index, (1*6*split_num), true_index))

                        index += 1
