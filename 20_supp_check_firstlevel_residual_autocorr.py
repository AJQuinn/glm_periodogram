import os
import mne
import osl
import glob
import h5py
import sails
import pandas as pd

from anamnesis import obj_from_hdf5file
import matplotlib.pyplot as plt

import numpy as np
from dask.distributed import Client

from lemon_support import (lemon_make_blinks_regressor,
                           lemon_make_task_regressor,
                           lemon_make_bads_regressor,
                           quick_plot_eog_icas,
                           quick_plot_eog_epochs,
                           get_eeg_data,
                           lemon_create_heog,
                           lemon_ica, lemon_check_ica)

from glm_config import cfg

#%% ----------------------------------------------------------
# GLM-Prep

fbase = os.path.join(cfg['lemon_processed_data'], '{subj}_preproc_raw.fif')
st = osl.utils.Study(fbase)

fname = st.get(subj='subj-010060')

runname = fname.split('/')[-1].split('.')[0]
print('processing : {0}'.format(runname))

subj_id = osl.utils.find_run_id(fname)

raw = mne.io.read_raw_fif(fname, preload=True)

icaname = fname.replace('preproc_raw.fif', 'ica.fif')
ica = mne.preprocessing.read_ica(icaname)

picks = mne.pick_types(raw.info, eeg=True, ref_meg=False)
chlabels = np.array(raw.info['ch_names'], dtype=h5py.special_dtype(vlen=str))[picks]

#%% ----------------------------------------------------------
# GLM-Prep

# Make blink regressor
fout = os.path.join(outdir, '{subj_id}_blink-summary.png'.format(subj_id=subj_id))
blink_vect, numblinks, evoked_blink = lemon_make_blinks_regressor(raw, figpath=fout)

fout = os.path.join(outdir, '{subj_id}_icaeog-summary.png'.format(subj_id=subj_id))
quick_plot_eog_icas(raw, ica, figpath=fout)

fout = os.path.join(outdir, '{subj_id}_{0}.png'.format('{0}', subj_id=subj_id))
quick_plot_eog_epochs(raw, figpath=fout)

veog = raw.get_data(picks='ICA-VEOG')[0, :]**2
veog = veog > np.percentile(veog, 97.5)

heog = raw.get_data(picks='ICA-HEOG')[0, :]**2
heog = heog > np.percentile(heog, 97.5)

# Make task regressor
task = lemon_make_task_regressor({'raw': raw})

# Make bad-segments regressor
bads_raw = lemon_make_bads_regressor(raw, mode='raw')
bads_diff = lemon_make_bads_regressor(raw, mode='diff')

# Get data
XX = get_eeg_data(raw).T
print(XX.shape)

# Run GLM-Periodogram
conds = {'Eyes Open': task == 1, 'Eyes Closed': task == -1}
covs = {'Linear Trend': np.linspace(0, 1, raw.n_times)}
confs = {'Bad Segments': bads_raw,
         'Bad Segments Diff': bads_diff,
         'V-EOG': veog, 'H-EOG': heog}
conts = [{'name': 'Mean', 'values':{'Eyes Open': 0.5, 'Eyes Closed': 0.5}},
         {'name': 'Open < Closed', 'values':{'Eyes Open': 1, 'Eyes Closed': -1}}]

fs = raw.info['sfreq']

npersegs = np.array([fs/5, fs/2, fs, fs*2, fs*5])

#%% ----------------------------------------------------------
# GLM-Prep

xc = []
dw = []

for ii in range(len(npersegs)):
    # Full model
    freq_vect, copes, varcopes, extras = sails.stft.glm_periodogram(XX, axis=0,
                                                                    fit_constant=False,
                                                                    conditions=conds,
                                                                    covariates=covs,
                                                                    confounds=confs,
                                                                    contrasts=conts,
                                                                    nperseg=int(npersegs[ii]),
                                                                    fmin=0.1, fmax=100,
                                                                    fs=fs, mode='magnitude',
                                                                    fit_method='glmtools')
    model, design, data = extras

    resids = extras[0].get_residuals(extras[2].data)

    xc_iter = np.zeros((resids.shape[1], 61))
    dw_iter = np.zeros((resids.shape[1], 61))

    for ii in range(resids.shape[1]):
        print(ii)
        for jj in range(61):
            #o = plt.xcorr(resids[:, ii, jj], resids[:, ii, jj], maxlags=2)
            xc_iter[ii, jj] = np.corrcoef(resids[1:, ii, jj], resids[:-1, ii, jj])[1, 0]
            dw_iter[ii, jj] = durbin_watson(resids[:, ii, jj])


