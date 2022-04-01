import os
import mne
import osl
import glob
import sails

import numpy as np
import h5py
import pandas as pd
from scipy import stats
import glmtools as glm

def run_first_level(fname, outdir):
    runname = fname.split('/')[-1].split('.')[0]
    print('processing : {0}'.format(runname))

    subj_id = osl.preprocessing.find_run_id(fname)[:12]

    raw = mne.io.read_raw_fif(fname, preload=True)

    picks = mne.pick_types(raw.info, eeg=True, ref_meg=False)
    chlabels = np.array(raw.info['ch_names'], dtype=h5py.special_dtype(vlen=str))[picks]

    #%% ----------------------------------------------------------
    # GLM-Prep

    # Make blink regressor
    fout = os.path.join(outdir, '{subj_id}_blink-summary.png'.format(subj_id=subj_id))
    blink_vect, numblinks, evoked_blink = lemon_make_blinks_regressor(raw, figpath=fout)

    # No reason to square if binarising?
    veog = raw.get_data(picks='ICA-VEOG')[0, :]**2
    thresh = np.percentile(veog, 95)
    veog = (veog>thresh).astype(float)

    heog = raw.get_data(picks='ICA-HEOG')[0, :]**2
    thresh = np.percentile(heog, 95)
    heog = (heog>thresh).astype(float)

    # Make bad-segments regressor
    bads = lemon_make_bads_regressor({'raw': raw})

    # Make ECG regressor
    ecg_ev, ecg_ch, ecg_avg = mne.preprocessing.find_ecg_events(dataset['raw'])
    heart_rate = np.zeros_like(veog, dtype=float)
    for ii in range(ecg_ev.shape[0]-1):
        start = ecg_ev[ii, 0]
        stop = ecg_ev[ii+1, 0]
        heart_rate[start:stop] = dataset['raw'].info['sfreq']/(stop-start)*60
    
    hbads = sails.utils.detect_artefacts(heart_rate, reject_mode='segments',
                                         segment_len=500, axis=0,
                                         gesd_args={'alpha': 0.1})
    heart_rate[hbads] = np.nan
    heart_rate[heart_rate<40] = np.nan
    heart_rate[np.isnan(heart_rate)] = np.nanmean(heart_rate)

    # Get data
    XX = raw.get_data(picks='grad').T
    XX = stats.zscore(XX, axis=0)
    print(XX.shape)
    # Run GLM-Periodogram
    covs = {'Heart Rate': heart_rate,
            'Linear': np.linspace(0, 1, len(heart_rate))}

    cons = {'H-EOG': heog, 'V-EOG': veog, 'Bad Segments': bads}
    fs = raw.info['sfreq']
    freq_vect, copes, varcopes, extras = sails.stft.glm_periodogram(XX, axis=0,
                                                                    covariates=covs,
                                                                    confounds=cons,
                                                                    nperseg=int(fs*2),
                                                                    fmin=0.1, fmax=100,
                                                                    fs=fs, mode='magnitude',
                                                                    fit_method='glmtools')
    model, design, data = extras
    data.info['dim_labels'] = ['Windows', 'Frequencies', 'Sensors']

    hdfname = os.path.join(outdir, '{subj_id}_glm-data.hdf5'.format(subj_id=subj_id))
    if os.path.exists(hdfname):
        print('Overwriting previous results')
        os.remove(hdfname)
    with h5py.File(hdfname, 'w') as F:
        model.to_hdf5(F.create_group('model'))
        design.to_hdf5(F.create_group('design'))
        data.to_hdf5(F.create_group('data'))
        F.create_dataset('freq_vect', data=freq_vect)
        F.create_dataset('chlabels', data=chlabels)
        F.create_dataset('scan_duration', data=raw.times[-1])
        F.create_dataset('num_blinks', data=numblinks)

    fout = os.path.join(outdir, '{subj_id}_glm-design.png'.format(subj_id=subj_id))
    design.plot_summary(show=False, savepath=fout)
    fout = os.path.join(outdir, '{subj_id}_glm-efficiency.png'.format(subj_id=subj_id))
    design.plot_efficiency(show=False, savepath=fout)

#%% -----------------------------------------------

from lemon_support import (lemon_make_blinks_regressor,
                           lemon_make_task_regressor,
                           lemon_make_bads_regressor,
                           lemon_set_channel_montage,
                           camcan_ica, lemon_check_ica)
from glm_config import cfg
extra_funcs = [camcan_ica]


treename = os.path.join('camcan.tree')
camcan = osl.utils.StudyTree(treename, cfg['camcan_raw'])
infiles = sorted(camcan.get('trans'))
print(len(infiles))

# Make some sensible output names
outnames = []
for infile in infiles:
    start = infile.find('sub-CC') 
    subj_id = infile[start:start+12]
    outnames.append(f'{subj_id}_transdef_mf2pt2_rest_raw.fif')

inputs = [(infiles[ii], outnames[ii]) for ii in range(len(infiles))]

config = osl.preprocessing.load_config('camcan_preproc.yml')

dataset = osl.preprocessing.run_proc_chain(inputs[0][0], config, outdir=cfg['camcan_preprocessed_data_dir'], extra_funcs=extra_funcs, outname=inputs[0][1])

#goods = osl.preprocessing.run_proc_batch(config, inputs[33:], proc_outdir, overwrite=True, nprocesses=3, extra_funcs=extra_funcs)

#%% ----------------------------------------------

glob_path = os.path.join(cfg['camcan_preprocessed_data_dir'], 'sub-*_preproc_raw.fif'))
proc_outputs = sorted(glob.glob(glob_path))

for fname in proc_outputs:
    try:
        run_first_level(fname, cfg['camcan_glm_data_dir'])
    except Exception as e:
        print(e)
        pass

