import os
import osl
import sys
import mne
import h5py
import dill
import sails
import pprint
import numpy as np
import glmtools as glm
from copy import deepcopy
from scipy import io, ndimage, stats
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

sys.path.append('/Users/andrew/src/qlt')
import qlt

import logging
logger = logging.getLogger('osl')

from glm_config import cfg

outdir = cfg['lemon_analysis_dir']

#%% --------------------------------------------------
# Preprocessing

from lemon_support import (lemon_make_blinks_regressor,
                           lemon_make_task_regressor,
                           lemon_make_bads_regressor,
                           lemon_set_channel_montage,
                           lemon_ica, lemon_check_ica,
                           lemon_create_heog, plot_design)
config = osl.preprocessing.load_config('lemon_preproc.yml')
pprint.pprint(config)

# 010003 010010 010050 010060 010089
subj_id = '010060'
mode = 'full'

base = '/Users/andrew/Projects/lemon/EEG_Raw_BIDS_ID/sub-{subj_id}/RSEEG'.format(subj_id=subj_id)
infile = os.path.join(base, 'sub-{subj_id}.vhdr'.format(subj_id=subj_id))
extras = [lemon_set_channel_montage, lemon_ica, lemon_create_heog]
dataset = osl.preprocessing.run_proc_chain(infile, config,
                                           extra_funcs=extras,
                                           outname='sub-{subj_id}_proc-full'.format(subj_id=subj_id),
                                           outdir=outdir,
                                           overwrite=True)

fout = os.path.join(outdir, 'sub-{subj_id}_proc-full_flowchart.png'.format(subj_id=subj_id))
osl.preprocessing.plot_preproc_flowchart(config)
plt.savefig(fout, dpi=300, transparent=True)
fout = os.path.join(outdir, 'sub-{subj_id}_proc-full_icatopos.png'.format(subj_id=subj_id))
lemon_check_ica(dataset, fout)

#%% ----------------------------------------------------

# GLM-Prep

# Make blink regressor
fout = os.path.join(outdir, 'sub-{subj_id}_proc-{mode}_blink-summary.png'.format(subj_id=subj_id, mode=mode))
blink_vect, numblinks, evoked_blink = lemon_make_blinks_regressor(dataset['raw'], figpath=fout)

veog = dataset['raw'].get_data(picks='ICA-VEOG')[0, :]**2
thresh = np.percentile(veog, 97.5)
veog = veog>thresh

heog = dataset['raw'].get_data(picks='ICA-HEOG')[0, :]**2
thresh = np.percentile(heog, 97.5)
heog = heog>thresh

# Make task regressor
task = lemon_make_task_regressor(dataset)

# Make bad-segments regressor
bads_raw = lemon_make_bads_regressor(dataset, mode='raw')
bads_diff = lemon_make_bads_regressor(dataset, mode='diff')

#bads[blink_vect>0] = 0

#%% --------------------------------------------------------
# GLM

raw = dataset['raw']
rawref = dataset['raw'].copy().pick_types(eeg=True)

XX = raw.get_data(picks='eeg').T
XX = stats.zscore(XX, axis=0)

conds = {'Eyes Open': task == 1, 'Eyes Closed': task == -1}
covs = {'Linear Trend': np.linspace(0, 1, dataset['raw'].n_times)}
confs = {'Bad Segments': bads_raw, 'Bad Segments Diff': bads_diff, 'V-EOG': veog, 'H-EOG': heog}
conts = [{'name': 'Mean', 'values':{'Eyes Open': 0.5, 'Eyes Closed': 0.5}},
         {'name': 'Open < Closed', 'values':{'Eyes Open': 1, 'Eyes Closed': -1}}]
fs = dataset['raw'].info['sfreq']
freq_vect, copes, varcopes, extras = sails.stft.glm_periodogram(XX, axis=0,
                                                                fit_constant=False,
                                                                conditions=conds,
                                                                covariates=covs,
                                                                confounds=confs,
                                                                contrasts=conts,
                                                                nperseg=int(fs*2),
                                                                fmin=0.1, fmax=100,
                                                                fs=fs, mode='magnitude',
                                                                fit_method='glmtools')
model, design, data = extras
data.info['dim_labels'] = ['Windows', 'Frequencies', 'Sensors']

hdfname = os.path.join(outdir, 'sub-{subj_id}_proc-{mode}_glm-data.hdf5'.format(subj_id=subj_id, mode=mode))
if os.path.exists(hdfname):
    print('Overwriting previous results')
    os.remove(hdfname)
with h5py.File(hdfname) as F:
     model.to_hdf5(F.create_group('model'))
     design.to_hdf5(F.create_group('design'))
     data.to_hdf5(F.create_group('data'))

fout = os.path.join(outdir, 'sub-{subj_id}_proc-{mode}_glm-design.png'.format(subj_id=subj_id, mode=mode))
design.plot_summary(show=False, savepath=fout)
fout = os.path.join(outdir, 'sub-{subj_id}_proc-{mode}_glm-efficiency.png'.format(subj_id=subj_id, mode=mode))
design.plot_efficiency(show=False, savepath=fout)

#%% ---------------------------------------------------------

plt.figure()
for ii in range(9):
    plt.subplot(2,5,ii+1)
    plt.plot(model.copes[ii, :, ])
    plt.title(model.contrast_names[ii])


#%% ---------------------------------------------------------
# Single channel example figure 1

sensor = 'Cz'

# Extract data segment
inds = np.arange(140*fs, 160*fs)
inds = np.arange(770*fs, 790*fs).astype(int)
inds = np.arange(520*fs, 550*fs).astype(int)

XX = stats.zscore(dataset['raw'].get_data(picks=sensor)[:, inds].T, axis=0)
mini_task = task[inds]
eog = dataset['raw'].copy().pick_types(eog=True, eeg=False)
eog.filter(l_freq=1, h_freq=25, picks='eog')
eog = eog.get_data(picks='eog')[:, inds].T
time = np.linspace(0, XX.shape[0]/fs, XX.shape[0])

# Compute STFT for data segment
config = sails.stft.PeriodogramConfig(input_len=XX.shape[0],
                                      fs=fs, nperseg=int(fs*2),
                                      axis=0, fmin=0.1, fmax=100)
config_flat = sails.stft.PeriodogramConfig(input_len=XX.shape[0],
                                           fs=fs, nperseg=int(fs*2),
                                           axis=0, fmin=0.1, fmax=100,
                                           window_type='boxcar')

condss = {'Eyes Open': (task == 1)[inds], 'Eyes Closed': (task == -1)[inds]}
covss = {'Linear Trend': np.linspace(0, 1, len(inds))}
confss = {'Bad Segments': bads_raw[inds], 'Bad Segments Diff': bads_diff[inds],
          'V-EOG': veog[inds], 'H-EOG': heog[inds]}
contss = [{'name': 'Mean', 'values':{'Eyes Open': 0.5, 'Eyes Closed': 0.5}},
         {'name': 'Open < Closed', 'values':{'Eyes Open': 1, 'Eyes Closed': -1}}]

# Compute mini-model to get design matrix
f, copes, varcopes, extras_s = sails.stft.glm_periodogram(XX, axis=0,
                                                          conditions=condss,
                                                          covariates=covss,
                                                          confounds=confss,
                                                          contrasts=contss,
                                                          fit_constant=False,
                                                          nperseg=int(fs*2),
                                                          fmin=0.1, fmax=100,
                                                          fs=fs, mode='magnitude',
                                                          fit_method='glmtools')
model_short, design_short, data_short = extras_s
stft = data_short.data[:, :, 0]
stft_time = np.arange(config.nperseg/2, XX.shape[0] - config.nperseg/2 + 1,
                      config.nperseg - config.noverlap)/float(fs)


# prep sqrt(f) axes
fx, ftl, ft = qlt.prep_scaled_freq(0.5, f)


#%% ------------------------------------------------------------
# Make figure 2

wlagX = sails.stft.apply_sliding_window(XX[:, 0],**config.sliding_window_args)
lagX = sails.stft.apply_sliding_window(XX[:, 0],**config_flat.sliding_window_args)

panel_label_height = 1.075
plt.figure(figsize=(16, 9))

ax_ts = plt.axes([0.05, 0.1, 0.4, 0.8])
ax_tf = plt.axes([0.45, 0.1, 0.16, 0.8])
#ax_tf_cb = plt.axes([0.62, 0.15, 0.01, 0.2])
ax_tf_cb = plt.axes([0.47, 0.065, 0.12, 0.01])
ax_des = plt.axes([0.7, 0.1, 0.25, 0.8])
ax_des_cb = plt.axes([0.96, 0.15, 0.01, 0.2])

ax_ts.plot(0.25*XX[:, 0], time, 'k', lw=0.5)
for ii in range(stft.shape[0]):
    jit = np.remainder(ii,3) / 5
    ax_ts.plot((2+jit, 2+jit), (ii, ii+2), lw=4)
    #ax_ts.plot((2+jit, 14), (ii+1, ii+1), lw=0.5, color=[0.8, 0.8, 0.8])
    ax_ts.plot((0, 14), (ii+1, ii+1), lw=0.5, color=[0.8, 0.8, 0.8])

ax_ts.set_prop_cycle(None)
x = np.linspace(4, 8, 500)
ax_ts.plot(x, 0.2*lagX.T + np.arange(stft.shape[0])[None, :] + 1, lw=0.8)

ax_ts.set_prop_cycle(None)
x = np.linspace(9, 13, 500)
ax_ts.plot(x, 0.2*wlagX.T + np.arange(stft.shape[0])[None, :] + 1, lw=0.8)
ax_ts.set_ylim(0, stft.shape[0]+1)
ax_ts.set_xlim(-2, 14)

for tag in ['top', 'right', 'bottom']:
    ax_ts.spines[tag].set_visible(False)
ax_ts.set_xticks([])
ax_ts.set_yticks(np.linspace(0,stft.shape[0]+1,7))
ax_ts.set_ylabel('Time (seconds)')

qlt.subpanel_label(ax_ts, 'A', xf=-0.02, yf=panel_label_height)
ax_ts.text(0.1, panel_label_height, '\nRaw EEG\nChannel: Cz', ha='center', transform=ax_ts.transAxes, fontsize='large')
qlt.subpanel_label(ax_ts, 'B', xf=0.25, yf=panel_label_height)
ax_ts.text(0.4, panel_label_height, 'Segmented EEG', ha='center', transform=ax_ts.transAxes, fontsize='large')
qlt.subpanel_label(ax_ts, 'C', xf=0.7, yf=panel_label_height)
ax_ts.text(0.825, panel_label_height, "'Windowed' EEG", ha='center', transform=ax_ts.transAxes, fontsize='large')

pcm = ax_tf.pcolormesh(fx, stft_time, stft, cmap='magma_r')
ax_tf.set_xticks(ft)
ax_tf.set_xticklabels(ftl)
plt.colorbar(pcm, cax=ax_tf_cb, orientation='horizontal')
ax_tf_cb.set_title('Magnitude')
for tag in ['bottom', 'right']:
    ax_tf.spines[tag].set_visible(False)
#ax_tf.yaxis.tick_right()
ax_tf.xaxis.tick_top()
ax_tf.set_xlabel('Frequency (Hz)')
ax_tf.xaxis.set_label_position('top')
ax_tf.set_yticks(np.linspace(0,stft.shape[0]+1,7))
ax_tf.set_yticklabels([])
ax_tf.text(0.475, panel_label_height, 'Short Time Fourier Tranform', va='center', ha='center', transform=ax_tf.transAxes, fontsize='large')
qlt.subpanel_label(ax_tf, 'D', xf=-0.05, yf=panel_label_height)

pcm = plot_design(ax_des, design_short.design_matrix, design.regressor_names)
for ii in range(len(design.regressor_names)):
    ax_des.text(0.5+ii, stft.shape[0], design.regressor_names[ii], ha='left', va='bottom', rotation=20)
ax_des.set_yticks(np.linspace(0,stft.shape[0]+1,7)-0.5, np.linspace(0, stft.shape[0]+1, 7).astype(int))
ax_des.text(0.25, panel_label_height, 'GLM Design Matrix', ha='center', transform=ax_des.transAxes, fontsize='large')
plt.colorbar(pcm, cax=ax_des_cb)
qlt.subpanel_label(ax_des, 'E', xf=-0.02, yf=panel_label_height)

fout = os.path.join(outdir, 'sub-{subj_id}_single-channel_glm-top.png'.format(subj_id=subj_id))
plt.savefig(fout, dpi=300, transparent=True)


#%% ---------------------------------------------------------

tstat_args = {'hat_factor': 5e-3, 'varcope_smoothing': 'medfilt', 'window_size': 15, 'smooth_dims': 1}
tstat_args2 = {'hat_factor': 5e-3, 'varcope_smoothing': 'medfilt', 'window_size': 15, 'smooth_dims': 0}

ch_ind = mne.pick_channels(raw.ch_names, ['Cz'])[0]
data_cz = deepcopy(data)
data_cz.data = data.data[:, :, ch_ind]

con_ind = [1, 7]

P1 = glm.permutations.ClusterPermutation(design, data_cz, con_ind[0], 500,
                                         pooled_dims=[1],
                                         tstat_args=tstat_args,
                                         cluster_forming_threshold=3,
                                         metric='tstats')
# BLINKS
P2 = glm.permutations.ClusterPermutation(design, data_cz, con_ind[1], 500,
                                         pooled_dims=[1],
                                         tstat_args=tstat_args,
                                         cluster_forming_threshold=3,
                                         metric='tstats')


fig = plt.figure(figsize=(16, 6))
ax = plt.axes([0.075, 0.2, 0.25, 0.6])
ax.plot(fx, model.copes[2, :, 0], label='Eyes Open')
ax.plot(fx, model.copes[3, :, 0], label='Eyes Closed')
ax.set_xticks(ft)
ax.set_xticklabels(ftl)
ax.set_ylabel('FFT Magnitude')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylim(0)
qlt.subpanel_label(ax, 'F', xf=-0.02, yf=1.1)
plt.legend(frameon=False)
ax.text(0.5, 1.1, 'GLM beta-spectrum', ha='center', transform=ax.transAxes, fontsize='large')
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)

for ii in range(len(con_ind)):
    plt.axes([0.4+ii*0.3, 0.7, 0.1, 0.15])
    plt.plot(fx, model.copes[con_ind[ii], :, 0])
    qlt.subpanel_label(plt.gca(), chr(71+ii), xf=-0.02, yf=1.3)
    plt.xticks(ft[::2], ftl[::2])
    for tag in ['top', 'right']:
        plt.gca().spines[tag].set_visible(False)
    plt.title('cope-spectrum')
    plt.axes([0.55+ii*0.3, 0.7, 0.1, 0.15])
    plt.plot(fx, model.varcopes[con_ind[ii], :, 0])
    plt.xticks(ft[::2], ftl[::2])
    for tag in ['top', 'right']:
        plt.gca().spines[tag].set_visible(False)
    plt.title('varcope-spectrum')

    ax = plt.axes([0.4+ii*0.3, 0.1, 0.25, 0.5])
    P = P1 if ii == 0 else P2
    clu = P.get_sig_clusters(data_cz, 99)
    for idx, c in enumerate(clu[1]):
        tinds = np.where(clu[0]==idx+1)[0]
        ax.axvspan(fx[tinds[0]], fx[tinds[-1]], facecolor=[0.7, 0.7, 0.7], alpha=0.5)
    ts = glm.fit.get_tstats(model.copes[con_ind[ii], :, ch_ind], model.varcopes[con_ind[ii], :, ch_ind], **tstat_args2)
    ax.plot(fx, ts)
    name = model.contrast_names[P.contrast_idx]
    ax.text(0.5, 1.7, f'Regressor: {name}', ha='center', transform=ax.transAxes, fontsize='large')
    ax.set_xticks(ft)
    ax.set_xticklabels(ftl)
    for tag in ['top', 'right']:
        plt.gca().spines[tag].set_visible(False)
    plt.title('t-spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('t-statistic')

fout = os.path.join(outdir, 'sub-{subj_id}_single-channel_glm-bottom.png'.format(subj_id=subj_id))
plt.savefig(fout, dpi=300, transparent=True)


#%% --------------------------------------------------------

adjacency, ch_names = mne.channels.channels._compute_ch_adjacency(raw.info, 'eeg')
ntests = np.prod(data.data.shape[1:])
ntimes = data.data.shape[1]
adjacency = mne.stats.cluster_level._setup_adjacency(adjacency, ntests, ntimes)

cft = 3
tstat_args = {'hat_factor': 5e-3, 'varcope_smoothing': 'medfilt', 'window_size': 15, 'smooth_dims': 1}

P = []
run_perms = False
for icon in range(1, design.num_contrasts):
    fpath = os.path.join(outdir, 'sub-{subj_id}_proc-{mode}_perms-con{icon}.pkl'.format(subj_id=subj_id, mode=mode, icon=icon))
    if run_perms:
        p = glm.permutations.MNEClusterPermutation(design, data, icon, 250,
                                                   nprocesses=3,
                                                   metric='tstats',
                                                   tstat_args=tstat_args,
                                                   cluster_forming_threshold=cft,
                                                   adjacency=adjacency)

        with open(fpath, "wb") as dill_file:
            dill.dump(p, dill_file)

        P.append(p)
    else:
        with open(fpath, 'rb') as dill_file:
            P.append(dill.load(dill_file))


#%% --------------------------------------------------------

ll = [['Rec Start', 'Rec End'],
      ['Good Seg', 'Bad Seg'],
      ['Good Seg', 'Bad Seg'],
      ['Low V-EOG Activity', 'High V-EOG Activity'],
      ['Low H-EOG Activity', 'High H-EOG Activity']]

col_heads = ['Mean', 'Linear Trend', 'Rest Condition', 'Bad Segments', 'VEOG', 'HEOG']
refraw = dataset['raw'].copy().pick_types(eeg=True)

plt.figure(figsize=(16, 16))
ax = plt.axes([0.075, 0.6, 0.175, 0.2])
ax.set_ylim(0, 0.0025)
qlt.plot_joint_spectrum(ax, model.copes[2, :, :], rawref, xvect=freq_vect,
                        freqs=[9], base=0.5, topo_scale=None,
                        ylabel='Amplitude', title=model.contrast_names[2])
qlt.subpanel_label(ax, chr(65), yf=1.6)

ax = plt.axes([0.3125, 0.6, 0.175, 0.2])
ax.set_ylim(0, 0.0025)
qlt.plot_joint_spectrum(ax, model.copes[3, :, :], rawref, xvect=freq_vect,
                        freqs=[9], base=0.5, topo_scale=None,
                        ylabel='Amplitude', title=model.contrast_names[3])
qlt.subpanel_label(ax, chr(66), yf=1.6)

ax = plt.axes([0.55, 0.6, 0.175, 0.3])
qlt.plot_sensorspace_clusters(data, P[0], rawref, ax, xvect=freq_vect,
                              ylabel='t-stat', base=0.5, topo_scale=None,
                              title=model.contrast_names[P[0].contrast_idx])
qlt.subpanel_label(ax, chr(67), yf=1.6)

ax = plt.axes([0.775, 0.6, 0.2, 0.2])
qlt.plot_channel_layout(ax, refraw, size=100)

for ii in range(5):
    ax = plt.axes([0.065+ii*0.195, 0.25, 0.125, 0.2])
    qlt.plot_sensorspace_clusters(data, P[ii+3], rawref, ax, xvect=freq_vect,
                                  ylabel='t-stat', base=0.5, topo_scale=None,
                                  title=model.contrast_names[P[ii+3].contrast_idx])
    qlt.subpanel_label(ax, chr(68+ii), yf=1.6)

    ax2 = plt.axes([0.065+ii*0.195, 0.07, 0.125, 0.2*2/3])
    ax2.set_ylim(0, 0.002)
    proj,llabels = model.project_range(P[ii+3].contrast_idx-2, nsteps=2)
    qlt.plot_sensor_data(ax2, proj.mean(axis=2).T, refraw, xvect=freq_vect, base=0.5, sensor_cols=False, lw=2)
    ylabel = 'Amplitude' if ii == 0 else ''
    qlt.decorate_spectrum(ax2, ylabel=ylabel)
    ax2.legend(ll[ii], frameon=False, fontsize=8)
    ax.set_title(model.contrast_names[P[ii+3].contrast_idx])

fout = os.path.join(outdir, 'sub-{subj_id}_proc-{mode}_glm-summary.png'.format(subj_id=subj_id, mode=mode))
plt.savefig(fout, dpi=300, transparent=True)

#%% --------------------------------------------------------

models = glm.fit.run_regressor_selection(design, data)

plt.figure(figsize=(16, 11))
plt.subplots_adjust(wspace=0.4, hspace=0.5, top=0.95, bottom=0.05)
labels = []

ref = models[0].r_square[0, :, :]
for ii in range(8):
    ax = plt.subplot(3, 4, ii+1)
    change =  models[ii].r_square[0, :, :] * 100
    qlt.plot_sensor_spectrum(ax, change, refraw, freq_vect, base=0.5)
    ax.set_ylabel('R-squared (%)')
    label = 'Full Model' if ii == 0 else "'{0}' only".format(models[0].regressor_names[ii-1])
    ax.set_title(label)
    labels.append(label)
    ax.set_ylim(0, 80)
    qlt.subpanel_label(ax, chr(65+ii))

ax = plt.subplot(313)
for ii in range(8):
    x = models[7-ii].r_square.flatten() * 100
    y = np.random.normal(ii+1, 0.05, size=len(x))
    plt.plot(x, y, 'r.', alpha=0.2)
h = plt.boxplot([m.r_square.flatten() * 100 for m in models[::-1]], vert=False, showfliers=False)
plt.yticks(np.arange(1,9),labels[::-1])
for tag in ['top', 'right']:
    ax.spines[tag].set_visible(False)
ax.set_xlabel('R-Squared (%)')
ax.set_ylabel('Model')
pos = list(ax.get_position().bounds)
pos[0] += 0.15
pos[2] -= 0.2
ax.set_position(pos)
qlt.subpanel_label(ax, chr(65+8))

fout = os.path.join(outdir, 'sub-{subj_id}_proc-{mode}_glm-modelselection.png'.format(subj_id=subj_id, mode=mode))
plt.savefig(fout, dpi=300, transparent=True)


#%% -------------------------------

freq_vect = f
nperseg = int(fs*2)
nstep = nperseg/2
noverlap = nperseg - nstep
time = np.arange(nperseg/2, dataset['raw'].n_times - nperseg/2 + 1,
                 nperseg - noverlap)/float(dataset['raw'].info['sfreq'])

#model, design, data = extras4

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
plt.pcolormesh(time, np.arange(len(model.regressor_names)), design.design_matrix[:, ::-1].T, cmap='RdBu_r')
plt.yticks(np.arange(len(model.regressor_names)), model.regressor_names[::-1])
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
plt.pcolormesh(time, freq_vect, fit.T, vmin=vmin, vmax=None, cmap='magma_r')
plt.title('Confound Regressors Only')
plt.colorbar()
qlt.subpanel_label(plt.gca(), 'D')

fout = os.path.join(outdir, 'sub-{subj_id}_single-channel_glm-singlechanTF.png'.format(subj_id=subj_id))
plt.savefig(fout, dpi=300, transparent=True)
