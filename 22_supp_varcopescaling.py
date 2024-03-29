import os
import mne
import osl
import glob
import h5py
import sails
import pandas as pd

from anamnesis import obj_from_hdf5file
import matplotlib.pyplot as plt
import glmtools as glm

import numpy as np
from dask.distributed import Client

import lemon_plotting
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

fname = st.get(subj='sub-010060')[0]

runname = fname.split('/')[-1].split('.')[0]
print('processing : {0}'.format(runname))

subj_id = osl.utils.find_run_id(fname)

raw = mne.io.read_raw_fif(fname, preload=True)
refraw = raw.copy().pick_types(eeg=True)

#%% ----------------------------------------------------------
# GLM-Prep

# Make blink regressor
blink_vect, numblinks, evoked_blink = lemon_make_blinks_regressor(raw, figpath=None)

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

#%% ----------------------------------------------------------
# GLM-Run


# Simple model
f, c, v, extras = sails.stft.glm_periodogram(XX, axis=0,
                                                 fit_constant=False,
                                                 conditions=conds,
                                                 covariates=covs,
                                                 confounds=confs,
                                                 contrasts=conts,
                                                 nperseg=int(fs*2),
                                                 noverlap=int(fs),
                                                 fmin=0.1, fmax=100,
                                                 fs=fs, mode='magnitude',
                                                 fit_method='glmtools')
model, design, data = extras

#%%


plt.figure(figsize=(16, 9))
plt.subplots_adjust(hspace=0.5, wspace=0.5)

ax = plt.subplot(2,3,1)
lemon_plotting.plot_sensor_spectrum(ax, model.copes[1, :, :], refraw, f, base=0.5)
ax.set_title('cope-spectrum')
lemon_plotting.subpanel_label(ax, 'A')

ax = plt.subplot(2,3,4)
lemon_plotting.plot_sensor_spectrum(ax, model.varcopes[1, :, :], refraw, f, base=0.5)
ax.set_title('varcope-spectrum')
lemon_plotting.subpanel_label(ax, 'B')

vc = glm.fit.varcope_corr_medfilt(model.varcopes[1, :, :], window_size=15, smooth_dims=0)
ax = plt.subplot(2,3,3)
lemon_plotting.plot_sensor_spectrum(ax, vc, refraw, f, base=0.5)
ax.set_title('smoothed varcope-spectrum')
lemon_plotting.subpanel_label(ax, 'E')

ax = plt.subplot(2,3,2)
ax.plot(np.abs(model.copes[1, :, :].reshape(-1)), model.varcopes[1, :, :].reshape(-1), '.')
lemon_plotting.subpanel_label(ax, 'C')
for tag in ['top', 'right']:
    ax.spines[tag].set_visible(False)
ax.set_xlabel('abs(cope_ values')
ax.set_ylabel('varcope values')

ax = plt.subplot(2,3,5)
lemon_plotting.plot_sensor_spectrum(ax, model.tstats[1, :, :], refraw, f, base=0.5)
ax.set_title('t-spectrum')
lemon_plotting.subpanel_label(ax, 'D')

ax = plt.subplot(2,3,6)
lemon_plotting.plot_sensor_spectrum(ax, model.copes[1, :, :]/np.sqrt(vc), refraw, f, base=0.5)
ax.set_title('varcope-smoothed t-spectrum')
lemon_plotting.subpanel_label(ax, 'F')

fout = os.path.join(cfg['lemon_figures'], 'lemon-supp_varcope-smoothing.png')
plt.savefig(fout, transparent=True, dpi=300)