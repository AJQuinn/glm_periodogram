import os
import sys
import osl
import numpy as np
import mne
import sails
import pprint
from scipy import io, ndimage, stats
import matplotlib.pyplot as plt

sys.path.append('/Users/andrew/src/qlt')
import qlt

import logging
logger = logging.getLogger('osl')

figbase = '/Users/andrew/Projects/glm/glm_psd/figures/'
outdir = '/Users/andrew/Projects/glm/glm_psd/analysis'

#%% --------------------------------------------------
# Preprocessing

from lemon_support import (lemon_make_blinks_regressor,
                           lemon_make_task_regressor,
                           lemon_make_bads_regressor,
                           lemon_set_channel_montage,
                           lemon_ica, lemon_check_ica)

config = osl.preprocessing.load_config('lemon_preproc.yml')
pprint.pprint(config)

outf = os.path.join(figbase, 'EEG_Lemon_preproc_flowchart.png')
osl.preprocessing.plot_preproc_flowchart(config, outname=outf, show=False, stagecol='lightskyblue', startcol='coral')

# Prep
subj = '010060'
base = f'/Users/andrew/Projects/lemon/EEG_Raw_BIDS_ID/sub-{subj}/RSEEG'
infile = os.path.join(base, f'sub-{subj}.vhdr')
extras = [lemon_set_channel_montage, lemon_ica]

# Run preproc with and without ICA
dataset_ica = osl.preprocessing.run_proc_chain(infile, config, extra_funcs=extras)
print(config['preproc'][8])
del config['preproc'][8]  # Drop ICA stage
pprint.pprint(config)
dataset_noica = osl.preprocessing.run_proc_chain(infile, config, extra_funcs=extras)

#%% ----------------------------------------------------------
# GLM-Prep

veog = dataset_ica['raw'].get_data(picks='ICA-VEOG')[0, :]**2
thresh = np.percentile(veog, 95)
veog = veog>thresh

heog = dataset_ica['raw'].get_data(picks='ICA-HEOG')[0, :]**2
thresh = np.percentile(heog, 95)
heog = heog>thresh

# Make task regressor
task = lemon_make_task_regressor(dataset_ica)

# Make bad-segments regressor
bads = lemon_make_bads_regressor(dataset_ica)

#bads[blink_vect>0] = 0

#%% --------------------------------------------------------
# GLM

raw = dataset_ica['raw']
XX = raw.get_data(picks='eeg').T
XX = stats.zscore(XX, axis=0)

covs = {'Linear Trend': np.linspace(0, 1, dataset_ica['raw'].n_times),
        'Eyes Open>Closed': task}
cons = {'Bad Segments': bads, 'V-EOG': veog, 'H-EOG': heog}
fs = dataset_ica['raw'].info['sfreq']


#%% ----------------------------------------------------------
# Now run four models
# 1 - no-ica - no confound
# 2 - no-ica + confound
# 3 - ica - no confound
# 4 - ica + confound

fs = int(dataset_noica['raw'].info['sfreq'])
inds = np.arange(247000, 396000) + fs*60  # about a minute of eyes open
inds = Ellipsis
sensor = 'Cz'
XX = stats.zscore(dataset_noica['raw'].get_data(picks=sensor)[:, inds].T, axis=0)

f, copes, varcopes, extras0 = sails.stft.glm_periodogram(XX, axis=0,
                                                         nperseg=fs*2,
                                                         fmin=0.1, fmax=45,
                                                         fs=fs, mode='magnitude',
                                                         fit_method='glmtools')
f, copes, varcopes, extras1 = sails.stft.glm_periodogram(XX, axis=0,
                                                         covariates=covs,
                                                         nperseg=fs*2,
                                                         fmin=0.1, fmax=45,
                                                         fs=fs, mode='magnitude',
                                                         fit_method='glmtools')
f, copes, varcopes, extras2 = sails.stft.glm_periodogram(XX, axis=0,
                                                         covariates=covs,
                                                         confounds=cons,
                                                         nperseg=fs*2,
                                                         fmin=0.1, fmax=45,
                                                         fs=fs, mode='magnitude',
                                                         fit_method='glmtools')
XX = stats.zscore(dataset_ica['raw'].get_data(picks=sensor)[:, inds].T, axis=0)
f, copes, varcopes, extras3 = sails.stft.glm_periodogram(XX, axis=0,
                                                         covariates=covs,
                                                         nperseg=fs*2,
                                                         fmin=0.1, fmax=45,
                                                         fs=fs, mode='magnitude',
                                                         fit_method='glmtools')
f, copes, varcopes, extras4 = sails.stft.glm_periodogram(XX, axis=0,
                                                         covariates=covs,
                                                         confounds=cons,
                                                         nperseg=fs*2,
                                                         fmin=0.1, fmax=45,
                                                         fs=fs, mode='magnitude',
                                                         fit_method='glmtools')


#%% ------------------------------------------------------------------
# Prep for figure 2

def plot_design(ax, design_matrix, regressor_names):
    num_observations, num_regressors = design_matrix.shape
    vm = np.max((design_matrix.min(), design_matrix.max()))
    cax = ax.pcolor(design_matrix, cmap=plt.cm.coolwarm,
                    vmin=-vm, vmax=vm)
    ax.set_xlabel('Regressors')
    tks = np.arange(len(regressor_names)+1)
    ax.set_xticks(tks+0.5)
    ax.set_xticklabels(tks)

    tkstep = 2
    tks = np.arange(0, design_matrix.shape[0], tkstep)

    for tag in ['top', 'right', 'left', 'bottom']:
        ax.spines[tag].set_visible(False)

    summary_lines = True
    new_cols = 0
    for ii in range(num_regressors):
        if summary_lines:
            x = design_matrix[:, ii]
            if np.abs(np.diff(x)).sum() != 0:
                #rn = np.max((np.abs(x.max()), np.abs(x.min())))
                #if (x.max() > 0) and (x.min() < 0):
                #    rn = rn*2
                #y = (x-x.min()) / (rn) * .8 + .1
                #y = (x) / (rn) * .8 + .1
                y = (0.5*x) / (np.max(np.abs(x)) * 1.1)
            else:
                # Constant regressor
                y = np.ones_like(x) * .45
            if num_observations > 50:
                ax.plot(y+ii+new_cols+0.5, np.arange(0, 0+num_observations)+.5, 'k')
            else:
                yy = y+ii+new_cols+0.5
                print('{} - {} - {}'.format(yy.min(), yy.mean(), yy.max()))
                ax.plot(y+ii+new_cols+0.5, np.arange(0, 0+num_observations)+.5,
                        'k|', markersize=5)

        # Add white dividing line
        if ii < num_regressors-1:
            ax.plot([ii+1+new_cols, ii+1+new_cols], [0, 0+num_observations],
                    'w', linewidth=4)
    return cax


def subpanel_label(ax, label, xf=-0.1, yf=1.1):
    ypos = ax.get_ylim()[0]
    yyrange = np.diff(ax.get_ylim())[0]
    ypos = (yyrange * yf) + ypos
    # Compute letter position as proportion of full xrange.
    xpos = ax.get_xlim()[0]
    xxrange = np.diff(ax.get_xlim())[0]
    xpos = (xxrange * xf) + xpos
    ax.text(xpos, ypos, label, horizontalalignment='center',
            verticalalignment='center', fontsize=20, fontweight='bold')


def decorate(ax):
    for tag in ['top', 'right']:
        ax.spines[tag].set_visible(False)


# Extract data segment
inds = np.arange(140*fs, 160*fs)
inds = np.arange(770*fs, 790*fs)

XX = stats.zscore(dataset_ica['raw'].get_data(picks=sensor)[:, inds].T, axis=0)
mini_task = task[inds]
eog = dataset_ica['raw'].copy().pick_types(eog=True, eeg=False)
eog.filter(l_freq=1, h_freq=25, picks='eog')
eog = eog.get_data(picks='eog')[:, inds].T
time = np.linspace(0, XX.shape[0]/fs, XX.shape[0])

# Compute STFT for data segment
config = sails.stft.PeriodogramConfig(input_len=XX.shape[0],
                                      fs=fs, nperseg=fs*2,
                                      axis=0, fmin=0.1,  fmax=45)
fw, tw, stft = sails.stft.compute_stft(XX[:, 0], **config.stft_args)

covs_short = {'Linear Trend': covs['Linear Trend'][inds],
              'Eyes Open>Closed': covs['Eyes Open>Closed'][inds]}
cons_short = {'Bad Segments': cons['Bad Segments'][inds],
              'V-EOG': cons['V-EOG'][inds],
              'H-EOG': cons['H-EOG'][inds]}

# Compute mini-model to get design matrix
f, copes, varcopes, extras5 = sails.stft.glm_periodogram(XX, axis=0,
                                                         covariates=covs_short,
                                                         confounds=cons_short,
                                                         nperseg=fs*2,
                                                         fmin=0.1, fmax=45,
                                                         fs=fs, mode='magnitude',
                                                         fit_method='glmtools')
model, design, data = extras5

# prep sqrt(f) axes
fx, ftl, ft = qlt.prep_scaled_freq(0.5, f)

#%% ------------------------------------------------------------
# Make figure 2

wlagX = sails.stft.apply_sliding_window(XX[:, 0],**config.sliding_window_args)
config.window = np.ones_like(config.window)
lagX = sails.stft.apply_sliding_window(XX[:, 0],**config.sliding_window_args)

plt.figure(figsize=(16, 9))

ax_ts = plt.axes([0.05, 0.1, 0.4, 0.8])
ax_tf = plt.axes([0.5, 0.1, 0.125, 0.8])
ax_tf_cb = plt.axes([0.62, 0.15, 0.01, 0.2])
ax_des = plt.axes([0.7, 0.1, 0.25, 0.8])
ax_des_cb = plt.axes([0.96, 0.15, 0.01, 0.2])

ax_ts.plot(0.25*XX[:, 0], time, 'k', lw=0.5)
for ii in range(19):
    jit = np.remainder(ii,3) / 5
    ax_ts.plot((2+jit, 2+jit), (ii, ii+2), lw=4)
    ax_ts.plot((2+jit, 13), (ii+1, ii+1), lw=0.5, color=[0.8, 0.8, 0.8])

ax_ts.set_prop_cycle(None)
x = np.linspace(4, 8, 500)
ax_ts.plot(x, 0.2*lagX.T + np.arange(19)[None, :] + 1, lw=0.8)

ax_ts.set_prop_cycle(None)
x = np.linspace(9, 13, 500)
ax_ts.plot(x, 0.2*wlagX.T + np.arange(19)[None, :] + 1, lw=0.8)
ax_ts.set_ylim(0, 20)

for tag in ['top', 'right', 'bottom']:
    ax_ts.spines[tag].set_visible(False)
ax_ts.set_xticks([])
ax_ts.set_yticks(np.linspace(0,20,5))
ax_ts.set_ylabel('Time (seconds)')

qlt.subpanel_label(ax_ts, 'A', xf=-0.02, yf=1.05)
ax_ts.text(0.1, 1.05, 'Raw EEG', ha='center', transform=ax_ts.transAxes, fontsize='large')
qlt.subpanel_label(ax_ts, 'B', xf=0.25, yf=1.05)
ax_ts.text(0.4, 1.05, 'Segmented EEG', ha='center', transform=ax_ts.transAxes, fontsize='large')
qlt.subpanel_label(ax_ts, 'C', xf=0.7, yf=1.05)
ax_ts.text(0.85, 1.05, "'Windowed' EEG", ha='center', transform=ax_ts.transAxes, fontsize='large')

pcm = ax_tf.pcolormesh(fx, tw, stft, cmap='magma_r')
ax_tf.set_xticks(ft)
ax_tf.set_xticklabels(ftl)
plt.colorbar(pcm, cax=ax_tf_cb)
ax_tf_cb.set_title('Magnitude', loc='left')
for tag in ['top', 'right']:
    ax_tf.spines[tag].set_visible(False)
ax_tf.set_xlabel('Frequency (Hz)')
ax_tf.set_yticks(np.linspace(0,20,5))
ax_tf.text(0.4, 1.05, 'Short Time\nFourier Tranform', va='center', ha='center', transform=ax_tf.transAxes, fontsize='large')
qlt.subpanel_label(ax_tf, 'D', xf=-0.05, yf=1.05)

pcm = plot_design(ax_des, design.design_matrix, design.regressor_names)
for ii in range(len(design.regressor_names)):
    ax_des.text(0.5+ii, 19, design.regressor_names[ii], ha='left', va='bottom', rotation=20)
ax_des.set_yticks(np.linspace(0,20,5)-0.5, np.linspace(0, 20, 5).astype(int))
ax_des.text(0.25, 1.05, 'GLM Design Matrix', ha='center', transform=ax_des.transAxes, fontsize='large')
plt.colorbar(pcm, cax=ax_des_cb)
qlt.subpanel_label(ax_des, 'E', xf=-0.02, yf=1.05)

fout = os.path.join(outdir, 'sub-{subj_id}_single-channel_glm-top.png'.format(subj_id=subj))
plt.savefig(fout, dpi=300, transparent=True)

#%% ---------------------------------------------------------

tstat_args = {'hat_factor': 5e-3, 'varcope_smoothing': 'medfilt', 'window_size': 15, 'smooth_dims': 1}

P1 = glm.permutations.ClusterPermutation(extras2[1], extras2[2], 2, 500,
                                         pooled_dims=[1],
                                         tstat_args=tstat_args,
                                         cluster_forming_threshold=3,
                                         metric='tstats')
# BLINKS
P2 = glm.permutations.ClusterPermutation(extras2[1], extras2[2], 4, 500,
                                         pooled_dims=[1],
                                         tstat_args=tstat_args,
                                         cluster_forming_threshold=3,
                                         metric='tstats')


fig = plt.figure(figsize=(16, 6))
ax = plt.axes([0.075, 0.2, 0.25, 0.6])
ax.plot(fx, extras2[0].copes[0, :, 0])
qlt.subpanel_label(ax, 'F', xf=-0.02, yf=1.1)
ax.text(0.5, 1.1, 'GLM-Spectrum', ha='center', transform=ax.transAxes, fontsize='large')

plt.axes([0.4, 0.7, 0.1, 0.15])
plt.plot(fx, extras2[0].copes[2, :, 0])
qlt.subpanel_label(plt.gca(), 'G', xf=-0.02, yf=1.3)
plt.axes([0.55, 0.7, 0.1, 0.15])
plt.plot(fx, extras2[0].varcopes[2, :, 0])

ax = plt.axes([0.4, 0.1, 0.25, 0.5])
clu = P1.get_sig_clusters(extras2[2], 95)
tinds = np.where(clu[0])[0]
ax.axvspan(fx[tinds[0]], fx[tinds[-1]], facecolor=[0.7, 0.7, 0.7], alpha=0.5)
ts = extras2[0].get_tstats(**tstat_args)[2, :, 0]
ax.plot(fx, ts)
ax.text(0.5, 1.6, 'Regressor-Spectrum\nEyes Open>Closed', ha='center', transform=ax.transAxes, fontsize='large')

plt.axes([0.7, 0.7, 0.1, 0.15])
plt.plot(fx, extras2[0].copes[4, :, 0])
qlt.subpanel_label(plt.gca(), 'H', xf=-0.02, yf=1.3)
plt.axes([0.855, 0.7, 0.1, 0.15])
plt.plot(fx, extras2[0].varcopes[4, :, 0])
ax = plt.axes([0.7, 0.1, 0.25, 0.5])
clu = P2.get_sig_clusters(extras2[2], 95)
for c in np.unique(clu[0]):
    if c == 0:
        continue
    tinds = np.where(clu[0]==c)[0]
    plt.axvspan(fx[tinds[0]], fx[tinds[-1]], facecolor=[0.7, 0.7, 0.7], alpha=0.5)
ts = extras2[0].get_tstats(**tstat_args)[4, :, 0]
plt.plot(fx, ts)
ax.text(0.5, 1.6, 'Regressor-Spectrum\nV-EOG', ha='center', transform=ax.transAxes, fontsize='large')

for idx, ax in enumerate(fig.axes):
    qlt.decorate_spectrum(ax)
    ax.set_xticks(ft)
    ax.set_xticklabels(ftl)
    if idx == 0:
        ax.set_ylabel('Magnitude')
    elif idx in [1, 4]:
        ax.set_ylabel('COPE')
        ax.set_xlabel('')
    elif idx in [2, 5]:
        ax.set_ylabel('VARCOPE')
        ax.set_xlabel('')
    elif idx in [3, 6]:
        ax.set_ylabel('t-stat')

fout = os.path.join(outdir, 'sub-{subj_id}_single-channel_glm-bottom.png'.format(subj_id=subj))
plt.savefig(fout, dpi=300, transparent=True)


#%% -------------------------------

freq_vect = f
nperseg = int(fs*2)
nstep = nperseg/2
noverlap = nperseg - nstep
time = np.arange(nperseg/2, dataset_ica['raw'].n_times - nperseg/2 + 1,
                 nperseg - noverlap)/float(dataset_ica['raw'].info['sfreq'])

model, design, data = extras4

vmin = 0
vmax = 0.0025
chan = 0

plt.figure(figsize=(16, 10))
plt.subplots_adjust(right=0.975, top=0.9, hspace=0.4)
plt.subplot(411)
plt.pcolormesh(time, freq_vect, data.data[:, :, chan].T, vmin=vmin, vmax=vmax, cmap='magma_r')
plt.xticks(np.arange(18)*60, np.arange(18))
plt.ylabel('Frequency (Hz)')
plt.title('STFT Data')
plt.colorbar()
qlt.subpanel_label(plt.gca(), 'A')

plt.subplot(412)
plt.pcolormesh(time, np.arange(6), design.design_matrix[:, ::-1].T, cmap='RdBu_r')
plt.yticks(np.arange(6), model.contrast_names[::-1])
plt.xticks(np.arange(18)*60, np.arange(18))
plt.title('Design Matrix')
plt.colorbar()
qlt.subpanel_label(plt.gca(), 'B')

regs = np.arange(3)
fit = np.dot(design.design_matrix[:, regs], model.betas[regs, :, chan])
plt.subplot(413)
plt.xticks(np.arange(18)*60, np.arange(18))
plt.ylabel('Frequency (Hz)')
plt.pcolormesh(time, freq_vect, fit.T, vmin=vmin, vmax=vmax, cmap='magma_r')
plt.title('Mean + Covariate Regressors')
plt.colorbar()
qlt.subpanel_label(plt.gca(), 'C')

regs = np.arange(3)+3
fit = np.dot(design.design_matrix[:, regs], model.betas[regs, :, chan])
plt.subplot(414)
plt.xticks(np.arange(18)*60, np.arange(18))
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (mins)')
plt.pcolormesh(time, freq_vect, fit.T, vmin=vmin, vmax=0.001, cmap='magma_r')
plt.title('Confound Regressors Only')
plt.colorbar()
qlt.subpanel_label(plt.gca(), 'D')

fout = os.path.join(outdir, 'sub-{subj_id}_single-channel_glm-singlechanTF.png'.format(subj_id=subj))
plt.savefig(fout, dpi=300, transparent=True)
